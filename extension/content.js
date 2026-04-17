/* ═══════════════════════════════════════════
   Hushly Guide Extension — Content Script
   Injected into every tab.
   Manages:
     1. Live Guide spotlight overlay (selector-first, path-memory aware)
     2. Page Assistant sidebar (floating chat panel)
   ═══════════════════════════════════════════ */

(function () {
  "use strict";

  if (window.__hushlyGuideInjected) return;
  window.__hushlyGuideInjected = true;

  // ══════════════════════════════════════════
  // LIVE GUIDE STATE
  // ══════════════════════════════════════════
  let guidePort    = null;
  let currentEl    = null;
  let currentStepIdx = 0;

  // ══════════════════════════════════════════
  // SIDEBAR STATE
  // ══════════════════════════════════════════
  let sidebarInjected = false;
  let sidebarVisible  = false;

  // ── CSS for the sidebar ───────────────────────────────────────────────────
  const SIDEBAR_CSS = `
    #hushly-sidebar-root * { box-sizing: border-box; }

    #hushly-sidebar-tab {
      position: fixed; right: 0; top: 50%; transform: translateY(-50%);
      z-index: 2147483630; width: 44px; height: 60px; background: #307fe2;
      border: none; border-radius: 14px 0 0 14px; cursor: pointer;
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 4px;
      box-shadow: -3px 0 16px rgba(48,127,226,0.35);
      transition: width 0.2s ease, background 0.2s ease; padding: 0;
    }
    #hushly-sidebar-tab:hover { width: 50px; background: #1a5db8; }
    #hushly-sidebar-tab svg { width: 20px; height: 20px; stroke: white; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; flex-shrink: 0; }
    #hushly-sidebar-tab .tab-dot { width: 5px; height: 5px; border-radius: 50%; background: rgba(255,255,255,0.6); }
    #hushly-sidebar-tab.open { background: #1a5db8; }
    #hushly-sidebar-tab.open .tab-dot { background: rgba(255,255,255,0.9); }

    #hushly-sidebar-panel {
      position: fixed; top: 0; right: -464px; width: 420px; height: 100vh;
      z-index: 2147483629; background: #f0f4fa;
      box-shadow: -6px 0 32px rgba(0,0,0,0.18);
      border-radius: 16px 0 0 16px; overflow: hidden;
      transition: right 0.32s cubic-bezier(0.4, 0, 0.2, 1);
    }
    #hushly-sidebar-panel.open { right: 0; border-radius: 0; }

    #hushly-sidebar-iframe { width: 100%; height: 100%; border: none; display: block; }

    #hushly-sidebar-resize {
      position: fixed; top: 0; right: 420px; width: 4px; height: 100vh;
      z-index: 2147483631; cursor: col-resize; background: transparent;
      transition: right 0.32s cubic-bezier(0.4, 0, 0.2, 1), background 0.15s; display: none;
    }
    #hushly-sidebar-panel.open ~ #hushly-sidebar-resize { display: block; }
    #hushly-sidebar-resize:hover { background: rgba(48,127,226,0.35); }
  `;

  // ══════════════════════════════════════════
  // MESSAGE LISTENERS
  // ══════════════════════════════════════════

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    switch (msg.type) {
      case "HUSHLY_PING":           sendResponse({ pong: true });          break;
      case "HUSHLY_TOGGLE_SIDEBAR": toggleSidebar(); sendResponse({ ok: true }); break;
      case "HUSHLY_HIDE_SIDEBAR":   hideSidebar();   sendResponse({ ok: true }); break;
      case "HUSHLY_SHOW_SIDEBAR":   showSidebar();   sendResponse({ ok: true }); break;
      case "HUSHLY_SCAN_PAGE":
        sendResponse({ elements: scanPage() });
        break;
    }
    return true;
  });

  chrome.runtime.onConnect.addListener((port) => {
    if (port.name !== "hushly-guide-port") return;
    guidePort = port;
    port.onMessage.addListener(handleGuidePortMessage);
    port.onDisconnect.addListener(() => { guidePort = null; removeHighlight(); });
  });

  // Safe port send — guards against "Extension context invalidated" after navigation
  function portSend(payload) {
    try {
      if (guidePort) guidePort.postMessage(payload);
    } catch (e) {
      // Context invalidated — port is dead, nothing we can do from this side
      guidePort = null;
    }
  }

  function handleGuidePortMessage(msg) {
    switch (msg.type) {
      case "HUSHLY_GUIDE_STEP":
        // Always rescan before each step — page may have navigated or modal may have opened
        scanPage();
        showStep(msg.step, msg.stepNum, msg.totalSteps, msg.pathId);
        break;
      case "HUSHLY_GUIDE_STOP":
        removeHighlight();
        removeDock();
        break;
    }
  }

  // ══════════════════════════════════════════
  // PAGE SCANNER — builds stable selectors
  // ══════════════════════════════════════════

  function buildSelector(el) {
    // Priority: id → data-testid → aria-label → data-nav/data-id → short path
    if (el.id) return `#${CSS.escape(el.id)}`;
    for (const attr of ["data-testid", "data-cy", "data-nav", "data-id", "data-name"]) {
      const val = el.getAttribute(attr);
      if (val) return `[${attr}="${val.replace(/"/g, '\\"')}"]`;
    }
    const aria = el.getAttribute("aria-label");
    if (aria) return `[aria-label="${aria.replace(/"/g, '\\"')}"]`;

    // Build a short nth-child path (max 3 levels, stop at body)
    const parts = [];
    let cur = el;
    for (let depth = 0; depth < 3 && cur && cur !== document.body; depth++) {
      const parent = cur.parentElement;
      if (!parent) break;
      const tag = cur.tagName.toLowerCase();
      const siblings = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
      if (siblings.length === 1) {
        parts.unshift(tag);
      } else {
        const nth = Array.from(parent.children).indexOf(cur) + 1;
        parts.unshift(`${tag}:nth-child(${nth})`);
      }
      cur = parent;
    }
    return parts.join(" > ") || el.tagName.toLowerCase();
  }

  function scanPage() {
    // Remove stale tags from previous scan
    document.querySelectorAll("[data-hushly-idx]").forEach(el =>
      el.removeAttribute("data-hushly-idx")
    );

    const seen    = new Set();
    const results = [];
    let   idx     = 0;

    // Very broad candidate query — SaaS apps rarely use semantic HTML
    const candidates = document.querySelectorAll(
      'button, a, [role="button"], [role="menuitem"], [role="tab"],' +
      '[role="link"], [role="option"], [role="treeitem"], [role="navigation"],' +
      '[role="menu"], [role="listitem"],' +
      'input:not([type="hidden"]), select, textarea,' +
      'nav *, aside li, aside a, aside [class*="item"],' +
      '[class*="nav-item"], [class*="NavItem"], [class*="menu-item"], [class*="MenuItem"],' +
      '[class*="menu-link"], [class*="nav-link"], [class*="sidebar-item"],' +
      '[class*="sidebar-link"], [class*="sidebarItem"], [class*="SidebarItem"],' +
      '[class*="sideNav"], [class*="side-nav"], [class*="leftNav"],' +
      '[class*="btn"], [class*="Btn"], [class*="button"], [class*="Button"],' +
      '[class*="link"], [class*="Link"], [class*="tab"], [class*="Tab"],' +
      '[tabindex]:not([tabindex="-1"]),' +
      'li[class], li[id], li[data-testid], li[onclick],' +
      '[onclick], [data-action], [data-href],' +
      'header *, footer *'
    );

    for (const el of candidates) {
      if (seen.has(el)) continue;

      // Use a looser visibility check for the scan —
      // collapsed-sidebar icons have 0-width text but the icon container is still there
      if (!hasArea(el)) continue;

      seen.add(el);
      el.setAttribute("data-hushly-idx", String(idx));
      const rect = el.getBoundingClientRect();

      // Get the most descriptive label we can find
      const label = getBestLabel(el);

      results.push({
        idx,
        tag:        el.tagName.toLowerCase(),
        text:       label,
        aria_label: el.getAttribute("aria-label") || "",
        placeholder: el.getAttribute("placeholder") || "",
        selector:   buildSelector(el),
        position:   { top: Math.round(rect.top + window.scrollY), left: Math.round(rect.left) }
      });
      idx++;
    }

    return results.slice(0, 150);
  }

  // Has a non-zero bounding box (less strict than isVisible — allows collapsed sidebars)
  function hasArea(el) {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  // Extract the most human-readable label for an element
  function getBestLabel(el) {
    const aria = el.getAttribute("aria-label") || el.getAttribute("title") || "";
    if (aria) return aria.trim().substring(0, 60);

    // Prefer text from direct children that are small (avoid capturing whole subtree)
    const directText = Array.from(el.childNodes)
      .filter(n => n.nodeType === Node.TEXT_NODE)
      .map(n => n.textContent.trim())
      .filter(Boolean)
      .join(" ");
    if (directText) return directText.substring(0, 60);

    // Look for an icon label span / tooltip span inside
    const labelEl = el.querySelector("[class*='label'], [class*='Label'], [class*='text'], span, .title");
    if (labelEl) {
      const t = (labelEl.textContent || "").trim();
      if (t && t.length < 60) return t;
    }

    // Fall back to full textContent (trimmed and deduped whitespace)
    return (el.textContent || "").replace(/\s+/g, " ").trim().substring(0, 60);
  }

  // ══════════════════════════════════════════
  // SIDEBAR — INJECT / SHOW / HIDE
  // ══════════════════════════════════════════

  function injectSidebar() {
    if (sidebarInjected) return;
    sidebarInjected = true;

    const style = document.createElement("style");
    style.id = "hushly-sidebar-style";
    style.textContent = SIDEBAR_CSS;
    (document.head || document.documentElement).appendChild(style);

    const root = document.createElement("div");
    root.id = "hushly-sidebar-root";

    const tab = document.createElement("button");
    tab.id = "hushly-sidebar-tab";
    tab.title = "Hushly Guide Assistant";
    tab.innerHTML = `
      <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      <div class="tab-dot"></div>`;
    tab.addEventListener("click", toggleSidebar);

    const panel = document.createElement("div");
    panel.id = "hushly-sidebar-panel";

    const iframe = document.createElement("iframe");
    iframe.id = "hushly-sidebar-iframe";
    iframe.src = chrome.runtime.getURL("popup.html") + "?sidebar=1";
    iframe.allow = "microphone";
    iframe.title = "Hushly Guide Assistant";

    panel.appendChild(iframe);
    root.appendChild(tab);
    root.appendChild(panel);
    document.body.appendChild(root);
  }

  function toggleSidebar() {
    if (!sidebarInjected) {
      injectSidebar();
      requestAnimationFrame(() => showSidebar());
    } else if (sidebarVisible) {
      hideSidebar();
    } else {
      showSidebar();
    }
  }

  function showSidebar() {
    if (!sidebarInjected) { injectSidebar(); requestAnimationFrame(showSidebar); return; }
    sidebarVisible = true;
    document.getElementById("hushly-sidebar-panel")?.classList.add("open");
    document.getElementById("hushly-sidebar-tab")?.classList.add("open");
  }

  function hideSidebar() {
    sidebarVisible = false;
    document.getElementById("hushly-sidebar-panel")?.classList.remove("open");
    document.getElementById("hushly-sidebar-tab")?.classList.remove("open");
  }

  // ══════════════════════════════════════════
  // PERSISTENT DOCK
  // ══════════════════════════════════════════

  let dockEl     = null;  // the persistent dock element
  let dockState  = {};    // current step data for rescan

  function ensureDock(text, hint, stepNum, totalSteps, pathId, selector, fallbacks) {
    dockState = { text, hint, stepNum, totalSteps, pathId, selector, fallbacks };

    if (!dockEl) {
      dockEl = document.createElement("div");
      dockEl.id = "hushly-floating-panel";
      dockEl.innerHTML = buildDockHTML(text, hint, stepNum, totalSteps);
      document.body.appendChild(dockEl);
      wireDockEvents(pathId, selector, fallbacks);
    } else {
      // Animate text update (vapor effect)
      const area = dockEl.querySelector(".hushly-dock-text-area");
      if (area) {
        area.classList.remove("vapor-in");
        area.classList.add("vapor-out");
        setTimeout(() => {
          updateDockText(text, hint, stepNum, totalSteps);
          area.classList.remove("vapor-out");
          area.classList.add("vapor-in");
          setTimeout(() => area.classList.remove("vapor-in"), 380);
        }, 240);
      }
      // Re-wire events with new data
      wireDockEvents(pathId, selector, fallbacks);
    }
  }

  function buildDockHTML(text, hint, stepNum, totalSteps) {
    return `
      <div class="hushly-dock-top">
        <div class="hushly-dock-meta-row">
          <span class="hushly-panel-badge">Hushly Guide</span>
          <span class="hushly-panel-step" id="hushly-dock-step">Step ${stepNum} of ${totalSteps}</span>
        </div>
        <button class="hushly-dock-close-btn" id="hushly-panel-close">✕</button>
        <div class="hushly-dock-text-area">
          <div class="hushly-panel-text"  id="hushly-dock-text">${esc(text)}</div>
          <div class="hushly-dock-hint"   id="hushly-dock-hint">${hint ? "Look for: " + esc(hint) : ""}</div>
        </div>
      </div>
      <div class="hushly-dock-bottom">
        <button class="hushly-dock-btn hushly-dock-btn-back"    id="hushly-back-btn">← Back</button>
        <span   class="hushly-dock-status"                      id="hushly-dock-status">Scanning…</span>
        <span   class="hushly-dock-sep"></span>
        <button class="hushly-dock-btn hushly-dock-btn-rescan"  id="hushly-rescan-page-btn">⟳ Rescan</button>
        <button class="hushly-dock-btn hushly-dock-btn-continue" id="hushly-done-btn">Next →</button>
        <button class="hushly-dock-btn hushly-dock-btn-wrong"   id="hushly-notfound-btn">✕ Wrong</button>
      </div>`;
  }

  function updateDockText(text, hint, stepNum, totalSteps) {
    const tEl  = dockEl?.querySelector("#hushly-dock-text");
    const hEl  = dockEl?.querySelector("#hushly-dock-hint");
    const sEl  = dockEl?.querySelector("#hushly-dock-step");
    if (tEl) tEl.textContent = text;
    if (hEl) hEl.textContent = hint ? "Look for: " + hint : "";
    if (sEl) sEl.textContent = `Step ${stepNum} of ${totalSteps}`;
  }

  function setDockStatus(msg, warn) {
    const el = dockEl?.querySelector("#hushly-dock-status");
    if (!el) return;
    el.textContent = msg;
    el.className   = "hushly-dock-status" + (warn ? " warn" : "");
  }

  function wireDockEvents(pathId, selector, fallbacks) {
    if (!dockEl) return;

    const closeBtn   = dockEl.querySelector("#hushly-panel-close");
    const rescanBtn  = dockEl.querySelector("#hushly-rescan-page-btn");
    const continueBtn= dockEl.querySelector("#hushly-done-btn");
    const wrongBtn   = dockEl.querySelector("#hushly-notfound-btn");
    const backBtn    = dockEl.querySelector("#hushly-back-btn");

    // Clone nodes to remove old listeners
    [closeBtn, rescanBtn, continueBtn, wrongBtn, backBtn].forEach(btn => {
      if (!btn) return;
      const clone = btn.cloneNode(true);
      btn.parentNode.replaceChild(clone, btn);
    });

    dockEl.querySelector("#hushly-panel-close")?.addEventListener("click", () => {
      removeDock(); removeHighlight();
      portSend({ type: "GUIDE_STOP_REQUESTED" });
    });

    dockEl.querySelector("#hushly-rescan-page-btn")?.addEventListener("click", () => {
      removeHighlight();
      setDockStatus("Rescanning…");
      scanPage();
      const { text, hint, stepNum, totalSteps } = dockState;
      tryShowStep(selector || "", fallbacks || [], hint, text, stepNum, totalSteps, pathId, false, 0);
    });

    dockEl.querySelector("#hushly-done-btn")?.addEventListener("click", () => {
      removeHighlight();
      portSend({ type: "STEP_COMPLETED" });
    });

    dockEl.querySelector("#hushly-notfound-btn")?.addEventListener("click", () => {
      const btn = dockEl?.querySelector("#hushly-notfound-btn");
      sendFeedback(pathId, (dockState.stepNum || 1) - 1, "element_not_found");
      if (btn) { btn.textContent = "✓ Reported"; btn.disabled = true; }
    });

    dockEl.querySelector("#hushly-back-btn")?.addEventListener("click", () => {
      removeHighlight();
      portSend({ type: "STEP_BACK" });
    });
  }

  function removeDock() {
    if (dockEl) { dockEl.remove(); dockEl = null; }
    dockState = {};
  }

  // ══════════════════════════════════════════
  // LIVE GUIDE — SHOW STEP
  // ══════════════════════════════════════════

  function showStep(step, stepNum, totalSteps, pathId) {
    removeHighlight(); // remove spotlight only — dock stays
    currentStepIdx = stepNum - 1;

    const selector  = step.selector || "";
    const fallbacks = step.selector_fallbacks || [];
    const text      = step.text || "";

    // If hint is empty, try to extract target label from the instruction text
    let hint = (step.element_hint || step.element_text || "").trim();
    if (!hint) hint = extractHintFromText(text);

    // Show / update the persistent dock
    ensureDock(text, hint, stepNum, totalSteps, pathId, selector, fallbacks);
    setDockStatus("Looking for element…");

    tryShowStep(selector, fallbacks, hint, text, stepNum, totalSteps, pathId, step.from_cache, 0);
  }

  // Extract likely UI label from step instruction text
  function extractHintFromText(text) {
    // Match: click/select/choose/tap/open + quoted or Title-Case word(s)
    const patterns = [
      /(?:click|select|choose|tap|open|find|navigate to|go to)\s+(?:the\s+|on\s+)?["""']([^"""']+)["""']/i,
      /(?:click|select|choose|tap|open|find)\s+(?:the\s+)?([A-Z][A-Za-z0-9 ]{1,30}?)(?:\s+(?:button|link|menu|tab|option|field|dropdown|item))?[.,\s]/i,
    ];
    for (const re of patterns) {
      const m = text.match(re);
      if (m?.[1]) return m[1].trim();
    }
    // Grab any double-quoted string
    const q = text.match(/["""']([^"""']{2,40})["""']/);
    if (q?.[1]) return q[1].trim();
    return "";
  }

  function tryShowStep(selector, fallbacks, hint, text, stepNum, totalSteps, pathId, fromCache, attempt) {
    const el = findBySelector(selector, fallbacks, hint);

    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      setDockStatus("👆 Click the highlighted element");
      setTimeout(() => highlightElement(el, text, hint, stepNum, totalSteps, pathId, fromCache, selector, fallbacks), 450);
      return;
    }

    // Retry with increasing delays — handles modals/dialogs that animate in
    if (attempt < 3) {
      const delay = [700, 1400, 2200][attempt];
      setDockStatus(`Waiting for element… (${attempt + 1}/3)`);
      setTimeout(() => {
        scanPage();
        tryShowStep(selector, fallbacks, hint, text, stepNum, totalSteps, pathId, fromCache, attempt + 1);
      }, delay);
      return;
    }

    // All retries exhausted — update dock status
    setDockStatus(hint ? `⚠ Could not find "${hint}" — use ⟳ Rescan` : "⚠ Element not found — use ⟳ Rescan", true);
    portSend({ type: "ELEMENT_NOT_FOUND", hint, step: text });
  }

  // ══════════════════════════════════════════
  // ELEMENT FINDING — selector-first
  // ══════════════════════════════════════════

  function findBySelector(selector, fallbacks, textHint) {
    // 1. Primary stored selector
    if (selector) {
      try {
        const el = document.querySelector(selector);
        if (el && isVisible(el)) return el;
      } catch {}
    }

    // 2. Fallback selectors
    for (const sel of (fallbacks || [])) {
      try {
        const el = document.querySelector(sel);
        if (el && isVisible(el)) return el;
      } catch {}
    }

    // 3. data-hushly-idx tag (set during current session scan)
    // (already consumed by selector if the scan was fresh)

    // 4. Text / aria fallback
    if (textHint) return findByText(textHint);

    return null;
  }

  function findByText(hint) {
    const h = hint.toLowerCase().trim();

    // Exact attribute matches
    for (const sel of [
      `[aria-label="${hint}"]`, `[aria-label="${h}"]`,
      `[title="${hint}"]`, `[placeholder="${hint}"]`,
    ]) {
      try {
        const el = document.querySelector(sel);
        if (el && isVisible(el)) return el;
      } catch {}
    }

    // Partial attribute matches
    const attrEls = document.querySelectorAll("[aria-label],[title],[placeholder],[data-testid]");
    for (const el of attrEls) {
      const val = (
        el.getAttribute("aria-label") || el.getAttribute("title") ||
        el.getAttribute("placeholder") || el.getAttribute("data-testid") || ""
      ).toLowerCase();
      if (val === h && isVisible(el)) return el;
    }
    for (const el of attrEls) {
      const val = (
        el.getAttribute("aria-label") || el.getAttribute("title") ||
        el.getAttribute("placeholder") || el.getAttribute("data-testid") || ""
      ).toLowerCase();
      if (val.includes(h) && isVisible(el)) return el;
    }

    // Interactive element text — exact match first
    const interactives = document.querySelectorAll(
      'button, a, [role="button"], [role="menuitem"], [role="option"], [role="tab"], [role="treeitem"], nav li, li[class]'
    );
    for (const el of interactives) {
      if ((el.textContent || "").trim().toLowerCase() === h && hasArea(el)) return el;
    }

    // Partial text match — prefer smallest element (most specific)
    const partial = [];
    for (const el of interactives) {
      const t = (el.textContent || "").trim().toLowerCase();
      if (t.includes(h) && hasArea(el)) {
        const r = el.getBoundingClientRect();
        partial.push({ el, area: r.width * r.height });
      }
    }
    if (partial.length) { partial.sort((a, b) => a.area - b.area); return partial[0].el; }

    // Word-level fallback — match if ALL words in the hint appear in the element text
    // Handles "Add Asset" vs "Add a New Asset" kind of variation
    const hWords = h.split(/\s+/).filter(w => w.length > 2);
    if (hWords.length >= 1) {
      const wordMatches = [];
      for (const el of interactives) {
        const t = (el.textContent || "").trim().toLowerCase();
        if (hWords.every(w => t.includes(w)) && hasArea(el)) {
          const r = el.getBoundingClientRect();
          wordMatches.push({ el, area: r.width * r.height });
        }
      }
      if (wordMatches.length) { wordMatches.sort((a, b) => a.area - b.area); return wordMatches[0].el; }
    }

    return null;
  }

  function isVisible(el) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    const s = window.getComputedStyle(el);
    return s.display !== "none" && s.visibility !== "hidden" && s.opacity !== "0";
  }

  // ══════════════════════════════════════════
  // HIGHLIGHT ELEMENT (spotlight + tooltip + arrow)
  // ══════════════════════════════════════════

  function highlightElement(el, text, hint, stepNum, totalSteps, pathId, fromCache, selector, fallbacks) {
    currentEl = el;
    const rect = el.getBoundingClientRect();

    // Spotlight border
    const highlight = document.createElement("div");
    highlight.id = "hushly-highlight";
    Object.assign(highlight.style, {
      top:    `${rect.top    - 5}px`,
      left:   `${rect.left   - 5}px`,
      width:  `${rect.width  + 10}px`,
      height: `${rect.height + 10}px`,
    });
    document.body.appendChild(highlight);

    // Pulse dot
    const dot = document.createElement("div");
    dot.id = "hushly-pulse-dot";
    dot.style.top  = `${rect.top  - 12}px`;
    dot.style.left = `${rect.right + 2}px`;
    document.body.appendChild(dot);

    // Bouncing arrow
    buildArrow(rect);

    // Tooltip with Close + Wrong + Rescan buttons
    document.body.appendChild(
      buildTooltip(text, hint, stepNum, totalSteps, rect, pathId, fromCache, selector, fallbacks)
    );

    // Click catcher
    const catcher = document.createElement("div");
    catcher.id = "hushly-click-catcher";
    Object.assign(catcher.style, {
      top:    `${rect.top}px`,
      left:   `${rect.left}px`,
      width:  `${rect.width}px`,
      height: `${rect.height}px`,
    });
    catcher.title = "Click to continue";
    catcher.addEventListener("click", onTargetClicked, { once: true });
    document.body.appendChild(catcher);
  }

  function buildArrow(rect) {
    const arrow = document.createElement("div");
    arrow.id = "hushly-arrow";

    const vh = window.innerHeight;
    const vw = window.innerWidth;
    const elCenterX = rect.left + rect.width / 2;
    const elCenterY = rect.top  + rect.height / 2;

    // Decide arrow direction based on where element is relative to viewport center
    const above = elCenterY < vh * 0.4;
    const below = elCenterY > vh * 0.6;
    const right = elCenterX > vw * 0.6;

    let arrowStyle, arrowChar;

    if (above) {
      // Element is in top portion — arrow points up from below
      arrowChar = "↑";
      Object.assign(arrow.style, {
        top:  `${rect.bottom + 14}px`,
        left: `${Math.min(Math.max(elCenterX - 18, 8), vw - 50)}px`,
      });
      arrow.dataset.dir = "up";
    } else if (below) {
      // Element is in bottom portion — arrow points down from above
      arrowChar = "↓";
      Object.assign(arrow.style, {
        top:  `${rect.top - 52}px`,
        left: `${Math.min(Math.max(elCenterX - 18, 8), vw - 50)}px`,
      });
      arrow.dataset.dir = "down";
    } else if (right) {
      // Element on right — arrow points right from left
      arrowChar = "→";
      Object.assign(arrow.style, {
        top:  `${Math.min(Math.max(elCenterY - 18, 8), vh - 50)}px`,
        left: `${rect.left - 56}px`,
      });
      arrow.dataset.dir = "right";
    } else {
      // Default — arrow points left from right
      arrowChar = "←";
      Object.assign(arrow.style, {
        top:  `${Math.min(Math.max(elCenterY - 18, 8), vh - 50)}px`,
        left: `${rect.right + 14}px`,
      });
      arrow.dataset.dir = "left";
    }

    arrow.textContent = arrowChar;
    document.body.appendChild(arrow);
  }

  function buildTooltip(text, hint, stepNum, totalSteps, rect, pathId, fromCache, selector, fallbacks) {
    const tooltip = document.createElement("div");
    tooltip.id = "hushly-tooltip";

    const spaceBelow = window.innerHeight - rect.bottom;
    const top  = spaceBelow >= 140 ? rect.bottom + 14 : Math.max(8, rect.top - 138);
    let   left = Math.max(8, rect.left);
    if (left > window.innerWidth - 350) left = window.innerWidth - 350;

    tooltip.style.top  = `${top}px`;
    tooltip.style.left = `${left}px`;

    const cacheBadge = fromCache
      ? `<span class="hushly-path-badge saved">✓ Saved Path</span>`
      : `<span class="hushly-path-badge ai">AI Generated</span>`;

    tooltip.innerHTML = `
      <div class="hushly-tooltip-header">
        <div class="hushly-step-num">Step ${stepNum} of ${totalSteps}</div>
        <div style="display:flex;align-items:center;gap:6px">
          ${cacheBadge}
          <button class="hushly-tooltip-close" id="hushly-close-btn" title="Stop guide">✕</button>
        </div>
      </div>
      <div class="hushly-step-text">${esc(text)}</div>
      <div class="hushly-step-hint"><span class="hushly-click-arrow">→</span> Click the highlighted element to continue</div>
      <div class="hushly-tooltip-footer">
        <button class="hushly-feedback-btn" id="hushly-wrong-btn">✕ Wrong element?</button>
        <button class="hushly-feedback-btn" id="hushly-rescan-btn" style="margin-left:6px">⟳ Rescan</button>
      </div>`;

    tooltip.addEventListener("click", (e) => {
      if (e.target.closest("#hushly-close-btn")) {
        e.stopPropagation();
        removeHighlight();
        portSend({ type: "GUIDE_STOP_REQUESTED" });
      } else if (e.target.closest("#hushly-wrong-btn")) {
        e.stopPropagation();
        sendFeedback(pathId, stepNum - 1, "wrong_element");
        showFeedbackConfirm(tooltip);
      } else if (e.target.closest("#hushly-rescan-btn")) {
        e.stopPropagation();
        removeHighlight();
        scanPage();
        tryShowStep(selector || "", fallbacks || [], hint, text, stepNum, totalSteps, pathId, fromCache, 0);
      }
    });

    return tooltip;
  }

  function sendFeedback(pathId, stepIdx, issue) {
    portSend({
      type:     "STEP_FEEDBACK",
      path_id:  pathId || "",
      step_idx: stepIdx,
      issue,
      page_url: window.location.href
    });
  }

  function showFeedbackConfirm(tooltip) {
    const btn = tooltip.querySelector("#hushly-wrong-btn");
    if (btn) {
      btn.textContent = "✓ Reported — thanks!";
      btn.disabled = true;
      btn.style.color = "#4ade80";
    }
  }

  function onTargetClicked() {
    try { if (currentEl) currentEl.click(); } catch {}
    removeHighlight();
    portSend({ type: "STEP_COMPLETED" });
  }

  // ══════════════════════════════════════════
  // FLOATING PANEL (element not found)
  // ══════════════════════════════════════════

  // ══════════════════════════════════════════
  // CLEANUP
  // ══════════════════════════════════════════

  function removeHighlight() {
    // Remove spotlight overlay only — dock stays alive
    ["hushly-highlight", "hushly-click-catcher", "hushly-tooltip",
     "hushly-pulse-dot", "hushly-arrow"].forEach(id => {
      document.getElementById(id)?.remove();
    });
    currentEl = null;
  }

  function esc(str) {
    return String(str)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Reposition on scroll/resize
  let repositionTimer;
  function repositionHighlight() {
    clearTimeout(repositionTimer);
    repositionTimer = setTimeout(() => {
      if (!currentEl) return;
      const rect = currentEl.getBoundingClientRect();
      const hl  = document.getElementById("hushly-highlight");
      const cc  = document.getElementById("hushly-click-catcher");
      const dot = document.getElementById("hushly-pulse-dot");
      if (hl)  { hl.style.top = `${rect.top - 5}px`; hl.style.left = `${rect.left - 5}px`; hl.style.width = `${rect.width + 10}px`; hl.style.height = `${rect.height + 10}px`; }
      if (cc)  { cc.style.top = `${rect.top}px`; cc.style.left = `${rect.left}px`; cc.style.width = `${rect.width}px`; cc.style.height = `${rect.height}px`; }
      if (dot) { dot.style.top = `${rect.top - 12}px`; dot.style.left = `${rect.right + 2}px`; }
    }, 100);
  }

  window.addEventListener("scroll", repositionHighlight, { passive: true });
  window.addEventListener("resize", repositionHighlight, { passive: true });

})();
