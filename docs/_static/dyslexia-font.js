(function () {
  const STORAGE_KEY = "neurosetta-font-mode";
  const LEGACY_KEY = "neurosetta-dyslexia-font";
  const MODES = ["default", "atkinson", "opendyslexic"];
  const LABELS = {
    default: "Default system font",
    atkinson: "Atkinson Hyperlegible",
    opendyslexic: "OpenDyslexic",
  };

  function normalizeMode(value) {
    if (MODES.includes(value)) {
      return value;
    }
    if (value === "on") {
      return "atkinson";
    }
    return "default";
  }

  function readStoredMode() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored !== null) {
        return normalizeMode(stored);
      }
      const legacy = localStorage.getItem(LEGACY_KEY);
      if (legacy !== null) {
        return normalizeMode(legacy);
      }
    } catch (e) {
      /* private browsing */
    }
    return "default";
  }

  function getMode() {
    return normalizeMode(document.body.dataset.fontMode || "default");
  }

  function setMode(mode) {
    const value = normalizeMode(mode);
    document.body.dataset.fontMode = value;
    try {
      localStorage.setItem(STORAGE_KEY, value);
      localStorage.removeItem(LEGACY_KEY);
    } catch (e) {
      /* private browsing */
    }
    syncButtons();
  }

  function cycleMode() {
    const idx = MODES.indexOf(getMode());
    setMode(MODES[(idx + 1) % MODES.length]);
  }

  function syncButtons() {
    const mode = getMode();
    document.querySelectorAll(".font-mode-toggle").forEach((btn) => {
      btn.dataset.fontMode = mode;
      btn.setAttribute("aria-pressed", mode === "default" ? "false" : "true");
      const label = `Font: ${LABELS[mode]}. Click to change.`;
      btn.setAttribute("aria-label", label);
      btn.title = label;
    });
  }

  function createButton() {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "font-mode-toggle";
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
      '<text class="font-mode-toggle-letter" x="12" y="17" text-anchor="middle" ' +
      'font-size="14" font-weight="700">Aa</text></svg>';
    btn.addEventListener("click", cycleMode);
    return btn;
  }

  function injectButtons() {
    document.querySelectorAll(".theme-toggle-container").forEach((container) => {
      if (container.querySelector(".font-mode-toggle")) {
        return;
      }
      container.appendChild(createButton());
    });
    syncButtons();
  }

  setMode(readStoredMode());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectButtons);
  } else {
    injectButtons();
  }
})();
