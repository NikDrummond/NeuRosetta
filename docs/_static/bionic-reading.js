(function () {
  const STORAGE_KEY = "neurosetta-bionic-reading";
  const SKIP_SELECTOR =
    "pre, code, kbd, samp, svg, script, style, textarea, .font-mode-toggle, .bionic-reading-toggle, .theme-toggle, .bionic-word";

  function isEnabled() {
    return document.body.dataset.bionicReading === "on";
  }

  function readStored() {
    try {
      return localStorage.getItem(STORAGE_KEY) === "on";
    } catch (e) {
      return false;
    }
  }

  function boldLength(word) {
    const len = word.length;
    if (len <= 1) {
      return len;
    }
    if (len <= 3) {
      return 1;
    }
    return Math.ceil(len / 2);
  }

  function shouldSkipElement(el) {
    return el.closest(SKIP_SELECTOR) !== null;
  }

  function bionicifyTextNode(textNode) {
    const parent = textNode.parentElement;
    if (!parent || shouldSkipElement(parent)) {
      return;
    }

    const text = textNode.nodeValue;
    if (!text || !/\S/.test(text)) {
      return;
    }

    const frag = document.createDocumentFragment();
    const parts = text.split(/(\s+)/);

    for (const part of parts) {
      if (!part) {
        continue;
      }
      if (/^\s+$/.test(part)) {
        frag.appendChild(document.createTextNode(part));
        continue;
      }

      const match = part.match(/^([^\w]*)(\w[\w'-]*)([^\w]*)$/);
      if (!match) {
        frag.appendChild(document.createTextNode(part));
        continue;
      }

      const [, lead, core, trail] = match;
      if (lead) {
        frag.appendChild(document.createTextNode(lead));
      }

      const span = document.createElement("span");
      span.className = "bionic-word";
      const split = boldLength(core);
      const strong = document.createElement("strong");
      strong.className = "bionic-bold";
      strong.textContent = core.slice(0, split);
      span.appendChild(strong);
      if (split < core.length) {
        span.appendChild(document.createTextNode(core.slice(split)));
      }
      frag.appendChild(span);

      if (trail) {
        frag.appendChild(document.createTextNode(trail));
      }
    }

    parent.replaceChild(frag, textNode);
  }

  function collectTextNodes(root) {
    const nodes = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || !/\S/.test(node.nodeValue)) {
          return NodeFilter.FILTER_REJECT;
        }
        const parent = node.parentElement;
        if (!parent || shouldSkipElement(parent)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    while (walker.nextNode()) {
      nodes.push(walker.currentNode);
    }
    return nodes;
  }

  function applyBionicReading() {
    const root = document.getElementById("furo-main-content");
    if (!root) {
      return;
    }
    collectTextNodes(root).forEach(bionicifyTextNode);
  }

  function removeBionicReading() {
    const root = document.getElementById("furo-main-content");
    if (!root) {
      return;
    }
    root.querySelectorAll(".bionic-word").forEach((span) => {
      span.replaceWith(document.createTextNode(span.textContent || ""));
    });
    root.normalize();
  }

  function syncButtons() {
    const enabled = isEnabled();
    document.querySelectorAll(".bionic-reading-toggle").forEach((btn) => {
      btn.setAttribute("aria-pressed", enabled ? "true" : "false");
      btn.classList.toggle("is-active", enabled);
      const label = enabled
        ? "Bionic reading on. Click to disable."
        : "Bionic reading off. Click to enable.";
      btn.setAttribute("aria-label", label);
      btn.title = label;
    });
  }

  function setEnabled(on, reprocess) {
    const value = on ? "on" : "off";
    document.body.dataset.bionicReading = value;
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {
      /* private browsing */
    }

    if (reprocess) {
      removeBionicReading();
      if (on) {
        applyBionicReading();
      }
    }
    syncButtons();
  }

  function toggle() {
    setEnabled(!isEnabled(), true);
  }

  function createButton() {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bionic-reading-toggle";
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
      '<text class="bionic-reading-toggle-label" x="12" y="17" text-anchor="middle" ' +
      'font-size="11" font-weight="700">BR</text></svg>';
    btn.addEventListener("click", toggle);
    return btn;
  }

  function injectButtons() {
    document.querySelectorAll(".theme-toggle-container").forEach((container) => {
      if (container.querySelector(".bionic-reading-toggle")) {
        return;
      }
      container.appendChild(createButton());
    });
    syncButtons();
  }

  function init() {
    injectButtons();
    if (isEnabled()) {
      applyBionicReading();
    }
    syncButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
