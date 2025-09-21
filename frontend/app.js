// --- element refs ---
const recordBtn      = document.getElementById('recordBtn');
const uploadBtn      = document.getElementById('uploadBtn');
const historyBtn     = document.getElementById('historyBtn');
const fileInput      = document.getElementById('fileInput');
const transcriptBox  = document.getElementById('transcriptBox'); // contenteditable="true"
const saveBtn        = document.getElementById('saveBtn');
const saveStatus     = document.getElementById('saveStatus');
const statusLine     = document.getElementById('statusLine');
const audioContainer = document.getElementById('audioContainer');
const moodRow        = document.getElementById('moodRow');
const moodBadge      = document.getElementById('moodBadge');
const emoChips       = document.getElementById('emoChips');

// Containers
const streakBadge    = document.getElementById('streakBadge');
const dashContainer  = document.getElementById('dashContainer');
const top1CountsEl   = document.getElementById('top1Counts');
const trendCanvas    = document.getElementById('trendSpark');
const searchInput    = document.getElementById('searchInput');
const detailsToggle  = document.getElementById('detailsToggle');
const emotionDetails = document.getElementById('emotionDetails');

// model selector
const modelSelect    = document.getElementById('modelSelect');
const modelNote      = document.getElementById('modelNote');

// images UI (hidden input + button + preview grid)
const imageInput     = document.getElementById('imageInput');       // hidden <input type="file" multiple>
const imageUploadBtn = document.getElementById('imageUploadBtn');   // “➕ Add images”
const imagePreview   = document.getElementById('imagePreview');     // preview grid before save

// images state (attached to the current entry being edited)
let pendingImages = []; // [{ filename, caption, tags }]

// 2k-char UI (emotion model analyzes first ~2000 chars)
const MAX_EMO_CHARS = 2000;
let charNote  = document.getElementById('charNote');
let charCount = document.getElementById('charCount');

// --- small UI helpers ---
function ensureCharCounterUI() {
  if (!transcriptBox) return;
  if (charNote && charCount) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'char-note-wrap';
  charNote = document.createElement('div');
  charNote.id = 'charNote';
  charNote.className = 'char-note';
  charNote.textContent = 'Only the first ~2,000 characters are analyzed for mood.';
  charCount = document.createElement('div');
  charCount.id = 'charCount';
  charCount.className = 'char-count';
  charCount.textContent = `0 / ${MAX_EMO_CHARS}`;
  wrapper.appendChild(charNote);
  wrapper.appendChild(charCount);
  if (transcriptBox.parentNode) {
    transcriptBox.parentNode.insertBefore(wrapper, transcriptBox.nextSibling);
  }
}
function getCurrentText() { return (transcriptBox?.innerText || '').trim(); }
function updateCharCounter() {
  ensureCharCounterUI();
  if (!charCount) return;
  const len = getCurrentText().length;
  charCount.textContent = `${len} / ${MAX_EMO_CHARS}`;
  if (len > MAX_EMO_CHARS) charCount.classList.add('warn'); else charCount.classList.remove('warn');
}

// --- global state ---
let mediaRecorder   = null;
let recordedChunks  = [];     // cumulative chunks while recording
let lastTranscribe  = null;   // { filename, transcript } for Save
let allEntries      = [];     // cached /entries list
let currentModel    = null;   // active whisper model (UI)

// live chunking (server keeps a rolling container)
let liveSessionId = null;
let livePartial   = [];       // reserved (not used by rolling approach)

// --- rendering helpers ---
function renderPlayer(filename) {
  audioContainer.innerHTML = '';
  if (!filename) return;
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.src = `/audio/${encodeURIComponent(filename)}`;
  audioContainer.appendChild(audio);
}
function clamp(n, lo, hi) { return Math.min(hi, Math.max(lo, n)); }
function moodToHue(mood) {
  const m = clamp(Number(mood || 0), -100, 100);
  return (m + 100) * 0.6; // map [-100,+100] → [0,120]
}
function renderMood(mood, top3) {
  if (mood === null || mood === undefined) {
    moodRow?.classList.add('is-hidden');
    if (emoChips) emoChips.innerHTML = '';
    return;
  }
  const hue = moodToHue(mood);
  if (moodBadge) {
    moodBadge.textContent = `Mood: ${mood > 0 ? '+' : ''}${mood}`;
    moodBadge.style.backgroundColor = `hsl(${hue} 20% 12%)`;
    moodBadge.style.borderColor     = `hsl(${hue} 20% 22%)`;
    moodBadge.style.color           = '#e8e8ea';
  }
  if (emoChips) {
    emoChips.innerHTML = '';
    (top3 || []).forEach(e => {
      const chip = document.createElement('span');
      const label = e.label || '';
      const score = typeof e.score === 'number' ? Math.round(e.score * 100) : null;
      chip.className = 'chip';
      chip.textContent = score !== null ? `${label} ${score}%` : label;
      emoChips.appendChild(chip);
    });
  }
  moodRow?.classList.remove('is-hidden');
}
function hideMood() { renderMood(undefined, undefined); }

// ===== Model selector =====
const ALLOWED_MODELS = ['tiny', 'base.en', 'small.en', 'medium.en'];
function populateModelSelect() {
  if (!modelSelect) return;
  if (modelSelect.options.length === 0) {
    ALLOWED_MODELS.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      modelSelect.appendChild(opt);
    });
  }
}
async function fetchBackendModelName() {
  try {
    const r = await fetch('http://127.0.0.1:8001/config/whisper_model');
    if (!r.ok) return null;
    const j = await r.json();
    return j?.name || null;
  } catch { return null; }
}
async function setBackendModelName(name) {
  const payload = { name };
  const r = await fetch('http://127.0.0.1:8001/config/whisper_model', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    let j = {}; try { j = await r.json(); } catch {}
    throw new Error(j.detail || `Failed to set model (${r.status})`);
  }
  return true;
}
async function initModelSelector() {
  if (!modelSelect) return;
  populateModelSelect();
  let initial = await fetchBackendModelName();
  if (!initial) initial = localStorage.getItem('whisperModel') || 'small.en';
  currentModel = initial; modelSelect.value = initial;
  if (modelNote) modelNote.textContent = `ASR model: ${initial}`;
  modelSelect.addEventListener('change', async () => {
    const next = modelSelect.value;
    if (next === currentModel) return;
    const prevStatus = statusLine?.textContent || '';
    statusLine.textContent = `Switching speech model to ${next}…`;
    try {
      await setBackendModelName(next);
      currentModel = next;
      localStorage.setItem('whisperModel', next);
      if (modelNote) modelNote.textContent = `ASR model: ${next}`;
      statusLine.textContent = `Model set to ${next}.`;
    } catch (e) {
      modelSelect.value = currentModel;
      statusLine.textContent = `❌ ${e.message}`;
    } finally {
      setTimeout(() => {
        if (statusLine.textContent.startsWith('Model set') || statusLine.textContent.startsWith('❌')) {
          statusLine.textContent = prevStatus;
        }
      }, 2000);
    }
  });
}

// ===== Date helpers =====
function parseLocalDateFromISOish(s) { if (!s) return null; const d = new Date(s); return isNaN(d) ? new Date() : d; }
function ymd(d) { const y = d.getFullYear(); const m = String(d.getMonth()+1).padStart(2,'0'); const da = String(d.getDate()).padStart(2,'0'); return `${y}-${m}-${da}`; }
function daysAgoDate(n) { const d = new Date(); d.setHours(0,0,0,0); d.setDate(d.getDate()-n); return d; }

// ===== Dashboard & Streaks =====
function computeWeeklyTop1Counts(items) {
  const start = daysAgoDate(6);
  const counts = {};
  items.forEach(it => {
    const d = parseLocalDateFromISOish(it.created_at);
    if (!d || d < start) return;
    const top = Array.isArray(it.emotions_top3) && it.emotions_top3[0];
    if (top && top.label) counts[top.label] = (counts[top.label] || 0) + 1;
  });
  return counts;
}
function drawTrendSparkline(items, canvas, maxPoints = 14) {
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 240;
  const h = canvas.height = canvas.clientHeight || 40;
  ctx.clearRect(0,0,w,h);
  const sorted = [...items].sort((a,b)=> (a.created_at||'').localeCompare(b.created_at||''));
  const last = sorted.slice(-maxPoints);
  if (last.length === 0) return;
  const vals = last.map(it => typeof it.mood === 'number' ? it.mood : 0);
  const min = -100, max = 100;
  const stepX = w / Math.max(1, last.length-1);
  ctx.lineWidth = 2; ctx.beginPath();
  last.forEach((_, i) => {
    const x = i * stepX;
    const v = vals[i];
    const y = h - ((v - min) / (max - min)) * h;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.strokeStyle = '#8dd3ff'; ctx.stroke();
  ctx.beginPath();
  const zeroY = h - ((0 - min)/(max-min))*h;
  ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY);
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.stroke();
}
function updateDashboard(items) {
  if (!dashContainer) return;
  const counts = computeWeeklyTop1Counts(items);
  if (top1CountsEl) {
    const parts = Object.entries(counts).sort((a,b)=> b[1]-a[1]).map(([label, n]) => `${label} ${n}`);
    top1CountsEl.textContent = parts.length ? parts.join(' · ') : 'No entries this week yet';
  }
  if (trendCanvas) drawTrendSparkline(items, trendCanvas, 14);
}
function computeStreakWithGrace(items) {
  // A “miss” grace of 1 day: streak breaks after 2 consecutive empty days.
  const set = new Set(items.map(it => ymd(parseLocalDateFromISOish(it.created_at))));
  let streak = 0, consecutiveMisses = 0;
  let d = new Date(); d.setHours(0,0,0,0);
  for (let i=0; i<365; i++) {
    const key = ymd(d);
    if (set.has(key)) { streak += 1; consecutiveMisses = 0; }
    else { consecutiveMisses += 1; if (consecutiveMisses >= 2) break; }
    d.setDate(d.getDate()-1);
  }
  return streak;
}
function updateStreakBadge(items) {
  if (!streakBadge) return;
  const s = computeStreakWithGrace(items);
  streakBadge.textContent = s > 0 ? `🔥 ${s}-day streak` : '🔥 0-day streak';
}

// ============================
// Upload a .wav/.mp3 file
// ============================
uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files?.[0];
  if (!file) { alert('Please select a .wav or .mp3 first.'); return; }
  statusLine.textContent = 'Uploading and transcribing…';
  transcriptBox.textContent = ''; audioContainer.innerHTML = '';
  hideMood(); updateCharCounter();

  try {
    const form = new FormData(); form.append('file', file);
    const res = await fetch('http://127.0.0.1:8001/transcribe/upload', { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
    const t = (data.transcript || '').trim();
    const lang = data.language || 'unknown';
    statusLine.textContent    = `Transcribed: ${data.filename} • Language: ${lang}`;
    transcriptBox.textContent = t; updateCharCounter();
    lastTranscribe = { filename: data.filename, transcript: t };
    renderPlayer(data.filename);
  } catch (e) {
    statusLine.textContent = '';
    transcriptBox.textContent = `❌ Error: ${e.message}`;
    updateCharCounter();
  }
});

// ============================
// Image upload / auto-tagging
// ============================
async function uploadOneImage(file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch('http://127.0.0.1:8001/images/upload', { method: 'POST', body: form });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || `Image upload failed (${r.status})`);
  return { filename: j.filename, caption: j.caption || null, tags: Array.isArray(j.tags) ? j.tags : [] };
}

function renderImagePreview() {
  if (!imagePreview) return;
  imagePreview.innerHTML = '';
  if (!pendingImages.length) {
    const small = document.createElement('small');
    small.className = 'muted';
    small.textContent = 'No images attached yet.';
    imagePreview.appendChild(small);
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'image-grid';
  pendingImages.forEach((im, idx) => {
    const card = document.createElement('div');
    card.className = 'image-card';

    const img = document.createElement('img');
    img.alt = im.caption || 'attachment';
    img.src = `/images/${encodeURIComponent(im.filename)}`;
    card.appendChild(img);

    if (im.caption) {
      const c = document.createElement('div');
      c.className = 'muted';
      c.style.fontSize = '11px';
      c.textContent = im.caption;
      card.appendChild(c);
    }

    const meta = document.createElement('div');
    meta.className = 'image-meta';
    (im.tags || []).forEach(t => {
      const tg = document.createElement('span');
      tg.className = 'tag';
      tg.textContent = t;
      meta.appendChild(tg);
    });
    card.appendChild(meta);

    const remove = document.createElement('button');
    remove.className = 'remove-btn';
    remove.textContent = 'Remove';
    remove.addEventListener('click', () => {
      pendingImages.splice(idx, 1);
      renderImagePreview();
    });
    card.appendChild(remove);

    grid.appendChild(card);
  });
  imagePreview.appendChild(grid);
}

// open file picker
imageUploadBtn?.addEventListener('click', () => {
  imageInput?.click();
});
// handle selected images
imageInput?.addEventListener('change', async () => {
  const files = Array.from(imageInput.files || []);
  if (!files.length) return;
  imageUploadBtn.disabled = true;
  imageUploadBtn.textContent = 'Adding…';
  try {
    for (const f of files) {
      try {
        const meta = await uploadOneImage(f);
        pendingImages.push(meta);
      } catch (e) {
        alert(`Image failed: ${e.message}`);
      }
    }
    renderImagePreview();
  } finally {
    imageUploadBtn.disabled = false;
    imageUploadBtn.textContent = '➕ Add images';
    imageInput.value = '';
  }
});

// ============================
// Save entry (text + images)
// ============================
saveBtn?.addEventListener('click', async () => {
  try {
    if (!lastTranscribe || !lastTranscribe.filename) {
      saveStatus.textContent = '⚠️ Nothing to save yet (upload or record first).';
      return;
    }
    const currentText = getCurrentText();
    const overLimit = currentText.length > MAX_EMO_CHARS;

    saveStatus.textContent = 'Saving…';
    const res = await fetch('http://127.0.0.1:8001/entries/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: lastTranscribe.filename,
        transcript: currentText,
        images: pendingImages.map(im => ({ filename: im.filename, caption: im.caption || null, tags: im.tags || [] })),
      }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || `Save failed (${res.status})`);

    saveStatus.textContent = `✔ Saved (id: ${j.id})` + (overLimit ? ' • analyzed first 2,000 chars' : '');

    // Refresh UI with server-evaluated mood and normalized image data
    try {
      const entry = await fetchEntry(j.id);
      renderMood(entry.mood, entry.emotions_top3);
      if (entry?.emotions_top3?.length) {
        const labels = entry.emotions_top3.map(e => e.label).join(' • ');
        statusLine.textContent = `Saved • Mood ${entry.mood > 0 ? '+' : ''}${entry.mood} • ${labels}` + (overLimit ? ' • (first 2,000 chars analyzed)' : '');
      }
      pendingImages = Array.isArray(entry.images)
        ? entry.images.map(im => ({ filename: im.filename, caption: im.caption || null, tags: im.tags || [] }))
        : [];
      renderImagePreview();
      await refreshHistoryCache();
    } catch (err) {
      console.warn('fetch entry after save failed:', err);
    }
  } catch (e) {
    saveStatus.textContent = `❌ Error: ${e.message}`;
  }
});

// ============================
// History + Search + Delete
// ============================
async function fetchEntries() {
  const r = await fetch('http://127.0.0.1:8001/entries');
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || `Failed to load history (${r.status})`);
  return j.items || [];
}
async function fetchEntry(id) {
  const r = await fetch(`http://127.0.0.1:8001/entries/${encodeURIComponent(id)}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || `Failed to load entry (${r.status})`);
  return j;
}
async function deleteEntry(id) {
  const r = await fetch(`http://127.0.0.1:8001/entries/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!r.ok) { let j = {}; try { j = await r.json(); } catch {} throw new Error(j.detail || `Delete failed (${r.status})`); }
  return true;
}
function fmt(dtString) { return dtString?.replace('T', ' ') || ''; }

const historyPanel = document.getElementById('historyPanel');

async function refreshHistoryCache() {
  allEntries = await fetchEntries();
  updateStreakBadge(allEntries);
  updateDashboard(allEntries);
  const q = (searchInput?.value || '').trim();
  await renderHistoryWithQuery(q);
}

// Ensure an entry is cached with transcript and derived image tags (for search)
async function ensureEntryCached(it) {
  if (typeof it._transcript !== 'string' || !Array.isArray(it._imageTags)) {
    const full = await fetchEntry(it.id);
    it._transcript = full.transcript || '';
    const tags = [];
    (full.images || []).forEach(im => {
      (im.tags || []).forEach(t => tags.push(String(t).toLowerCase()));
      if (im.caption) {
        tags.push(...String(im.caption).toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
      }
    });
    it._imageTags = Array.from(new Set(tags));
    it._emotions_all = full.emotions_all || null;
  }
  return it;
}

async function renderHistoryWithQuery(query) {
  if (!historyPanel) return;
  const items = allEntries;
  let listToShow = items;
  const q = (query || '').trim();

  if (q) {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    const results = [];
    for (const it of items) {
      await ensureEntryCached(it);
      const text = it._transcript.toLowerCase();
      const tags = (it._imageTags || []).join(' ');
      const ok = terms.every(t => text.includes(t) || tags.includes(t));
      if (ok) results.push(it);
    }
    listToShow = results;
  }
  renderHistoryList(listToShow);
}

function makeDeleteButton(it) {
  const del = document.createElement('button');
  del.className = 'btn btn-danger';
  del.textContent = 'Delete';
  del.addEventListener('click', async (ev) => {
    ev.stopPropagation();
    const yes = confirm('Delete this entry and associated audio?');
    if (!yes) return;
    try {
      await deleteEntry(it.id);
      await refreshHistoryCache();
      saveStatus.textContent = '🗑️ Entry deleted';
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  });
  return del;
}

function renderHistoryList(items) {
  historyPanel.classList.remove('is-hidden');
  if (!items.length) { historyPanel.textContent = 'No entries yet.'; return; }

  const list = document.createElement('ul');
  list.className = 'list';

  items.forEach((it) => {
    const li = document.createElement('li');
    li.className = 'list-item';

    const row = document.createElement('div');
    row.className = 'btn btn-block';

    const topLine = document.createElement('div');
    topLine.textContent =
      `${fmt(it.created_at)}  —  ${it.words ?? 0} words` +
      (it.image_count ? ` • 🖼️ ${it.image_count} image${it.image_count>1?'s':''}` : '');

    const sub = document.createElement('div');
    sub.style.marginTop = '4px';
    sub.style.display = 'flex';
    sub.style.alignItems = 'center';
    sub.style.gap = '8px';

    if (typeof it.mood === 'number') {
      const hue = moodToHue(it.mood);
      const badge = document.createElement('span');
      badge.className = 'mood-badge';
      badge.textContent = `Mood: ${it.mood > 0 ? '+' : ''}${it.mood}`;
      badge.style.backgroundColor = `hsl(${hue} 20% 12%)`;
      badge.style.borderColor     = `hsl(${hue} 20% 22%)`;
      sub.appendChild(badge);
    }

    if (Array.isArray(it.emotions_top3) && it.emotions_top3.length) {
      const chips = document.createElement('div');
      chips.className = 'chips';
      it.emotions_top3.forEach(e => {
        const c = document.createElement('span');
        c.className = 'chip';
        c.textContent = e.label;
        chips.appendChild(c);
      });
      sub.appendChild(chips);
    }

    sub.appendChild(makeDeleteButton(it));

    row.appendChild(topLine);
    row.appendChild(sub);

    row.addEventListener('click', async () => {
      try {
        statusLine.textContent = 'Loading entry…';
        transcriptBox.textContent = '';
        audioContainer.innerHTML = '';
        updateCharCounter();

        const entry = await fetchEntry(it.id);

        statusLine.textContent    = `🗂 ${fmt(entry.created_at)} (${entry.audio_filename})`;
        transcriptBox.textContent = entry.transcript || '';
        updateCharCounter();

        lastTranscribe = { filename: entry.audio_filename, transcript: entry.transcript || '' };
        renderPlayer(entry.audio_filename);

        renderMood(entry.mood, entry.emotions_top3);
        renderEmotionDetails(entry);

        // Load images attached to this entry into the editor preview
        pendingImages = Array.isArray(entry.images)
          ? entry.images.map(im => ({ filename: im.filename, caption: im.caption || null, tags: im.tags || [] }))
          : [];
        renderImagePreview();

      } catch (e) {
        statusLine.textContent = '';
        transcriptBox.textContent = `❌ ${e.message}`;
        updateCharCounter();
      }
    });

    li.appendChild(row);
    list.appendChild(li);
  });

  historyPanel.innerHTML = '';
  historyPanel.appendChild(list);
}

function renderEmotionDetails(entry) {
  if (!emotionDetails) return;
  const all = entry.emotions_all || {};
  const items = Object.entries(all).sort((a,b)=> b[1]-a[1]);
  emotionDetails.innerHTML = '';
  items.forEach(([label, p]) => {
    const line = document.createElement('div');
    const pct = Math.round((p||0)*100);
    line.textContent = `${label}: ${pct}%`;
    emotionDetails.appendChild(line);
  });
}
if (detailsToggle) {
  detailsToggle.addEventListener('click', () => {
    if (!emotionDetails) return;
    const hidden = emotionDetails.classList.toggle('is-hidden');
    detailsToggle.textContent = hidden ? 'Show details' : 'Hide details';
  });
}

historyBtn.addEventListener('click', async () => {
  try {
    historyPanel.classList.remove('is-hidden');
    historyPanel.textContent = 'Loading history…';
    await refreshHistoryCache();
  } catch (e) {
    historyPanel.textContent = `❌ ${e.message}`;
  }
});

// Live search in history list
if (searchInput) {
  searchInput.addEventListener('input', () => {
    const q = searchInput.value || '';
    renderHistoryWithQuery(q);
  });
}

// Keep counter live while editing transcript
['input', 'keyup', 'paste', 'cut'].forEach(ev => {
  transcriptBox?.addEventListener(ev, () => updateCharCounter());
});

// ===============================================
// Recording (cumulative upload + flush on stop)
// ===============================================

let pendingUploads = [];   // promises for /transcribe/chunk requests
let chunkCounter   = 0;    // debugging index

// Build a cumulative WebM blob, POST to /transcribe/chunk, show live text if any
async function uploadCumulativeBlob() {
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  if (!blob.size) return;

  const file = new File([blob], `upto-${Date.now()}.webm`, { type: 'audio/webm' });

  const form = new FormData();
  form.append('session_id', liveSessionId);
  form.append('index', String(chunkCounter++));
  form.append('file', file);

  const p = fetch('http://127.0.0.1:8001/transcribe/chunk', { method: 'POST', body: form })
    .then(r => r.json().then(j => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      if (!ok) throw new Error(j.detail || 'Chunk upload failed');
      const t = (j.transcript || '').trim();
      // Only update preview while still recording
      if (mediaRecorder && mediaRecorder.state === 'recording' && t) {
        statusLine.textContent = `🔴 Live transcribing… session ${liveSessionId.slice(0, 8)}`;
        transcriptBox.textContent = t;
        updateCharCounter();
      }
    })
    .catch(err => console.warn('chunk upload failed', err));

  pendingUploads.push(p);
  // Keep only the most recent N to cap memory
  if (pendingUploads.length > 8) {
    pendingUploads = pendingUploads.slice(-8);
  }
}

recordBtn.addEventListener('click', async () => {
  try {
    // ----- STOP: flush + wait + finalize -----
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      recordBtn.textContent = '● Record';
      statusLine.textContent = 'Finalizing…';

      // 1) force final dataavailable event
      const flushed = new Promise((resolve) => {
        mediaRecorder.addEventListener('dataavailable', () => resolve(), { once: true });
      });
      mediaRecorder.requestData();
      await flushed;

      // 2) stop and wait (release mic)
      const stopped = new Promise((resolve) => {
        mediaRecorder.addEventListener('stop', () => resolve(), { once: true });
      });
      mediaRecorder.stop();
      await stopped;

      // 3) final cumulative upload
      await uploadCumulativeBlob();

      // 4) wait for any in-flight chunk uploads
      if (pendingUploads.length) {
        await Promise.allSettled(pendingUploads);
      }

      // 5) finalize on server (transcribe full rolling container)
      const res = await fetch('http://127.0.0.1:8001/transcribe/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: liveSessionId }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || `Finalize failed (${res.status})`);

      statusLine.textContent    = `Final transcript (${j.words} words)`;
      transcriptBox.textContent = j.final_transcript || '';
      updateCharCounter();

      lastTranscribe = {
        filename: j.audio_filename || null,
        transcript: j.final_transcript || '',
      };
      renderPlayer(lastTranscribe.filename);

      // reset session state
      liveSessionId   = null;
      recordedChunks  = [];
      pendingUploads  = [];
      chunkCounter    = 0;
      return;
    }

    // ----- START: new recording session -----
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    pendingUploads = [];
    chunkCounter   = 0;
    liveSessionId  = crypto.randomUUID();

    statusLine.textContent    = `🔴 Live transcribing… session ${liveSessionId.slice(0, 8)}`;
    transcriptBox.textContent = '';
    audioContainer.innerHTML  = '';
    hideMood();
    updateCharCounter();

    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    // For each slice, add to cumulative buffer and upload in the background
    mediaRecorder.ondataavailable = async (e) => {
      if (!e.data || e.data.size === 0) return;
      recordedChunks.push(e.data);
      uploadCumulativeBlob(); // fire-and-forget; awaited on stop
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((tr) => tr.stop());
    };

    // Shorter slice size reduces “tail loss” if user stops mid-slice
    mediaRecorder.start(4000); // 4s timeslice
    recordBtn.textContent = '■ Stop';

  } catch (err) {
    statusLine.textContent = '';
    transcriptBox.textContent = `❌ Mic error: ${err.message}`;
    updateCharCounter();
  }
});

// ===== init on load =====
ensureCharCounterUI();
updateCharCounter();
initModelSelector();
renderImagePreview(); // show empty state initially
