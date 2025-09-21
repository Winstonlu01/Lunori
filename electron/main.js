// electron/main.js
const { app, BrowserWindow, session } = require("electron");

const APP_URL = "http://127.0.0.1:8001/";
const APP_ORIGIN = new URL(APP_URL).origin; // "http://127.0.0.1:8001"
const PARTITION = "persist:lunori";        // persistent Chromium storage

function registerPermissions() {
  const ses = session.fromPartition(PARTITION);

  // Say "yes" up-front for our own origin so Chromium does not prompt.
  ses.setPermissionCheckHandler((wc, permission, details) => {
    if (permission === "media" && details.securityOrigin === APP_ORIGIN) {
      return true;
    }
    return false;
  });

  // Fallback: if a prompt path still triggers, auto-grant for our origin.
  ses.setPermissionRequestHandler((wc, permission, callback, details) => {
    if (
      permission === "media" &&
      typeof details.requestingUrl === "string" &&
      details.requestingUrl.startsWith(APP_ORIGIN)
    ) {
      return callback(true);
    }
    callback(false);
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1100,
    height: 800,
    backgroundColor: "#0b0b0c",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      partition: PARTITION, // persistent session so grants stick
    },
  });

  // Harden navigation: only allow our own origin.
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (e, url) => {
    try {
      const dest = new URL(url);
      if (dest.origin !== APP_ORIGIN) e.preventDefault();
    } catch {
      e.preventDefault();
    }
  });

  win.loadURL(APP_URL);
}

app.whenReady().then(() => {
  registerPermissions();
  createWindow();
});

app.on("window-all-closed", () => {
  app.quit();
});
