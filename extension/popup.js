/* ═══════════════════════════════════════════
   Hushly Guide Extension — Popup Logic
   ═══════════════════════════════════════════ */

// ── Config ──────────────────────────────────
let BACKEND_URL = "http://localhost:8000";

// ── Detect display mode ──
const IS_SIDEBAR = new URLSearchParams(window.location.search).has("sidebar");
const IS_POPOUT  = new URLSearchParams(window.location.search).has("popout");

// ── DOM refs ────────────────────────────────
const messagesEl       = document.getElementById("messages");
const inputEl          = document.getElementById("input");
const emptyState       = document.getElementById("emptyState");
const sendBtn          = document.getElementById("sendBtn");
const enhanceBtn       = document.getElementById("enhanceBtn");
const micBtn           = document.getElementById("micBtn");
const pageContextBtn   = document.getElementById("pageContextBtn");
const chatBanner       = document.getElementById("chatBanner");
const statusDot        = document.getElementById("statusDotIndicator");
const statusText       = document.getElementById("statusTextWidget");
const visualizerEl     = document.getElementById("visualizer");
const suggestionContainer = document.getElementById("suggestionContainer");
const guideModeBar     = document.getElementById("guideModeBar");
const guideModeLabel   = document.getElementById("guideModeLabel");
const guideModeIcon    = document.getElementById("guideModeIcon");
const guideStepCounter = document.getElementById("guideStepCounter");

// ── App State ────────────────────────────────
// null = unknown (first load), true = connected, false = confirmed disconnected
let isServerConnected = null;
let chatHistory       = [];
let lastInputMethod   = "text";

// ── Guide State ──────────────────────────────
// guidedMode: null | 'voice' | 'live'
let guidedMode    = null;
let guidedData    = null;   // { task_title, steps: [{text, selector, element_hint}] }
let currentStep   = 0;
let inGuidePrompt = false;  // waiting for user to choose a guide type
let currentPathId = null;   // path ID of active guide (for feedback)
let lastQuery     = "";     // last successful user query
let lastAnswer    = "";     // last successful KB answer

// ── Port to content script (for live guide) ──
let contentPort = null;

// ── Voice State ──────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition  = null;
let isListening  = false;
let cachedVoices = [];

// ── Suggestions ─────────────────────────────
const SUGGESTION_POOL = [
  "How do I create an asset?",
  "How to upload a content asset?",
  "What is GEOSherpa?",
  "How to configure UTM tracking?",
  "How to create an Experience?",
  "Setup ABM Page",
  "What are Personas?",
  "How to integrate with Salesforce?",
  "What is a Content Stream?",
  "How do I add a visitor group?",
  "What is Hushly AEO?",
  "How to set up a campaign template?",
];

// ═══════════════════════════════════════════
// GUIDE SESSION PERSISTENCE
// ═══════════════════════════════════════════
function saveGuideSession() {
  if (typeof chrome === "undefined" || !chrome.storage) return;
  const state = guidedMode === "live" && guidedData ? {
    guidedMode, guidedData, currentStep,
    currentPathId, lastQuery, lastAnswer,
    chatHistory: chatHistory.slice(-10)
  } : null;
  chrome.storage.session.set({ hushlyGuideSession: state }).catch(() => {});
}

async function restoreGuideSession() {
  if (typeof chrome === "undefined" || !chrome.storage?.session) return;
  try {
    const { hushlyGuideSession: s } = await chrome.storage.session.get("hushlyGuideSession");
    if (!s || !s.guidedData || !s.guidedMode) return;

    // Restore state
    guidedMode    = s.guidedMode;
    guidedData    = s.guidedData;
    currentStep   = s.currentStep || 0;
    currentPathId = s.currentPathId || null;
    lastQuery     = s.lastQuery || "";
    lastAnswer    = s.lastAnswer || "";
    if (s.chatHistory?.length) chatHistory = s.chatHistory;

    hideEmpty();
    const step  = guidedData.steps[currentStep];
    const total = guidedData.steps.length;
    addBotMsg(
      `*(Guide resumed: **${esc(guidedData.task_title)}** — Step ${currentStep + 1} of ${total})*`
    );
    updateGuideModeBar();

    // Re-connect to content script and re-send current step
    await new Promise(r => setTimeout(r, 400));
    await connectContentPort();
    sendCurrentLiveStep();
  } catch (e) {
    console.warn("restoreGuideSession:", e);
  }
}

function clearGuideSession() {
  if (typeof chrome !== "undefined" && chrome.storage?.session) {
    chrome.storage.session.remove("hushlyGuideSession").catch(() => {});
  }
}

// ═══════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════
async function init() {
  // Apply display mode classes
  if (IS_SIDEBAR) {
    document.body.classList.add("sidebar-mode");
    const pinBtn = document.getElementById("pinBtn");
    if (pinBtn) { pinBtn.title = "Close page assistant"; pinBtn.classList.add("active"); }
  }

  if (IS_POPOUT) {
    document.body.classList.add("popout-mode");
    const bar = document.getElementById("popoutBar");
    if (bar) bar.style.display = "flex";
    // Hide the pin button (no point in sidebar from a popout)
    const pinBtn = document.getElementById("pinBtn");
    if (pinBtn) pinBtn.style.display = "none";
    // Hide the popout button itself (already popped out)
    const popoutBtn = document.getElementById("popoutBtn");
    if (popoutBtn) popoutBtn.style.display = "none";
    // Restore chat history from session storage
    await restorePopoutHistory();
    // Wire up window drag
    initPopoutDrag();
  }

  // Load saved backend URL
  if (typeof chrome !== "undefined" && chrome.storage) {
    const data = await chrome.storage.local.get(["backendUrl"]);
    if (data.backendUrl) {
      BACKEND_URL = data.backendUrl;
      const inp = document.getElementById("backendUrlInput");
      if (inp) inp.value = BACKEND_URL;
    }
  }

  // Sidebar opened after popout: restore chat history
  if (IS_SIDEBAR && !IS_POPOUT) {
    await restorePopoutHistory();
  }

  initSuggestions();
  initVoice();
  checkServerStatus();
  setInterval(checkServerStatus, 3000);
  await restoreGuideSession();
}

function initSuggestions() {
  const shuffled = [...SUGGESTION_POOL].sort(() => 0.5 - Math.random());
  const selected = shuffled.slice(0, 4);
  suggestionContainer.innerHTML = selected
    .map(q => `<button class="suggestion-chip" data-action="fill-input">${q}</button>`)
    .join("");
}

// ═══════════════════════════════════════════
// PAGE ASSISTANT (SIDEBAR MODE)
// ═══════════════════════════════════════════
async function openPageAssistant() {
  if (typeof chrome === "undefined") return;

  if (IS_SIDEBAR) {
    // We ARE the sidebar — tell content script to hide the panel
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab) await chrome.tabs.sendMessage(tab.id, { type: "HUSHLY_HIDE_SIDEBAR" }).catch(() => {});
    } catch {}
    return;
  }

  // ── Regular popup: open sidebar on the current page ──
  const pinBtn = document.getElementById("pinBtn");

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) { showPinError("No active tab found."); return; }

    // Blocked URLs (chrome:// etc.) cannot have content scripts
    if (tab.url?.startsWith("chrome://") || tab.url?.startsWith("chrome-extension://") ||
        tab.url?.startsWith("about:") || tab.url?.startsWith("edge://")) {
      showPinError("Open a regular web page first, then try again.");
      return;
    }

    // Try sending message to already-running content script
    const sent = await chrome.tabs.sendMessage(tab.id, { type: "HUSHLY_TOGGLE_SIDEBAR" })
      .catch(() => null);

    if (sent) {
      // Content script was already there — close popup and let sidebar show
      window.close();
      return;
    }

    // Content script not loaded yet (tab was open before extension was installed/reloaded)
    // → inject it now programmatically
    if (pinBtn) { pinBtn.title = "Injecting…"; pinBtn.style.opacity = "0.5"; }

    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["guide-overlay.css"] });

    // Give the script a moment to initialise its message listeners
    await new Promise(r => setTimeout(r, 300));

    await chrome.tabs.sendMessage(tab.id, { type: "HUSHLY_TOGGLE_SIDEBAR" });
    window.close();

  } catch (e) {
    console.error("openPageAssistant:", e);
    showPinError("Could not open sidebar: " + (e.message || e));
    if (pinBtn) { pinBtn.style.opacity = ""; }
  }
}

function showPinError(msg) {
  const pinBtn = document.getElementById("pinBtn");
  if (!pinBtn) return;
  pinBtn.style.opacity = "";
  // Flash the button red briefly with a tooltip
  pinBtn.title = msg;
  pinBtn.style.color = "#ef4444";
  pinBtn.style.borderColor = "#ef4444";
  setTimeout(() => {
    pinBtn.style.color = "";
    pinBtn.style.borderColor = "";
    pinBtn.title = "Open as page assistant";
  }, 2500);
  // Also show a brief inline message in the chat
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `
    <div class="bubble" style="background:#fef2f2;border-color:#fecaca;color:#dc2626;font-size:12.5px">
      ⚠️ ${esc(msg)}
    </div>`;
  messagesEl.appendChild(el);
  scroll();
}

// ═══════════════════════════════════════════
// SETTINGS
// ═══════════════════════════════════════════
function toggleSettings() {
  const panel = document.getElementById("settingsPanel");
  panel.style.display = panel.style.display === "none" ? "flex" : "none";
  if (panel.style.display !== "none") {
    document.getElementById("backendUrlInput").value = BACKEND_URL;
  }
}

async function saveSettings() {
  const val = document.getElementById("backendUrlInput").value.trim();
  if (val) {
    BACKEND_URL = val.replace(/\/$/, "");
    if (typeof chrome !== "undefined" && chrome.storage) {
      await chrome.storage.local.set({ backendUrl: BACKEND_URL });
    }
  }
  document.getElementById("settingsPanel").style.display = "none";
  checkServerStatus();
}

// ═══════════════════════════════════════════
// SERVER HEALTH CHECK
// ═══════════════════════════════════════════
async function checkServerStatus() {
  // Manual timeout (AbortSignal.timeout not reliable in all extension contexts)
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);

  try {
    const res = await fetch(`${BACKEND_URL}/health`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    if (isServerConnected !== true) {
      isServerConnected = true;
      setEnabled(true);
      chatBanner.classList.add("online");
      statusDot.className = "status-dot online";
      statusText.textContent = "Server connected";
    }
  } catch (err) {
    clearTimeout(timer);
    // Trigger on EVERY failure (null→false AND true→false), not just first
    if (isServerConnected !== false) {
      isServerConnected = false;
      setEnabled(false);
      showServerError();
    }
  }
}

function showServerError() {
  chatBanner.classList.remove("online");
  chatBanner.style.background = "rgba(239,68,68,0.08)";
  chatBanner.style.color = "#dc2626";
  chatBanner.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:6px;width:100%">
      <div style="display:flex;align-items:center;gap:8px;font-weight:600">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        Cannot reach server at <code style="background:rgba(220,38,38,0.12);padding:1px 5px;border-radius:4px;font-size:11px">${BACKEND_URL}</code>
      </div>
      <div style="font-size:11.5px;opacity:0.8">
        Start the backend: <code style="background:rgba(220,38,38,0.12);padding:1px 5px;border-radius:4px">python app.py</code>
        &nbsp;·&nbsp; Or change URL in <button data-action="toggle-settings" style="background:none;border:none;color:inherit;cursor:pointer;text-decoration:underline;font-size:inherit;padding:0">Settings ⚙</button>
      </div>
    </div>`;
  statusDot.className = "status-dot offline";
  statusText.textContent = "Not connected";
}

function setEnabled(on) {
  inputEl.disabled = !on;
  sendBtn.disabled = !on;
  enhanceBtn.disabled = !on;
  micBtn.disabled = !on;
  pageContextBtn.disabled = !on;
  inputEl.placeholder = on ? "Ask a question about Hushly…" : "Start the server to chat…";
}

// ═══════════════════════════════════════════
// VOICE (Web Speech API)
// ═══════════════════════════════════════════
function initVoice() {
  if (!SpeechRecognition) return;

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    isListening = true;
    micBtn.classList.add("listening");
    visualizerEl.classList.add("active");
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.classList.remove("listening");
    visualizerEl.classList.remove("active");
    // In voice guide mode, re-open mic after speech stops
    if (guidedMode === "voice" && !window.speechSynthesis?.speaking) {
      setTimeout(() => { if (!isListening) recognition.start(); }, 400);
    }
  };

  recognition.onerror = (e) => {
    if (e.error !== "aborted" && e.error !== "no-speech") {
      console.warn("Speech recognition error:", e.error);
    }
    isListening = false;
    micBtn.classList.remove("listening");
    visualizerEl.classList.remove("active");
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript.trim();
    if (guidedMode === "voice") {
      // Intercept guide commands directly; don't put them in input
      handleVoiceGuideCommand(transcript);
    } else if (inGuidePrompt) {
      handleGuidePromptVoice(transcript);
    } else {
      inputEl.value = transcript;
      lastInputMethod = "voice";
      send();
      lastInputMethod = "text";
    }
  };

  // Pre-load voices (Chrome bug fix — voices load async)
  loadVoices();
  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }
}

function loadVoices() {
  if (!window.speechSynthesis) return;
  const voices = window.speechSynthesis.getVoices();
  if (voices.length) cachedVoices = voices;
}

function getBestVoice() {
  if (!cachedVoices.length) cachedVoices = window.speechSynthesis?.getVoices() || [];
  // Prefer natural-sounding English voices
  const pref = [
    v => v.name.includes("Google") && v.name.includes("US English"),
    v => v.name.includes("Google") && v.name.toLowerCase().includes("english"),
    v => v.lang === "en-US",
    v => v.lang.startsWith("en"),
  ];
  for (const test of pref) {
    const found = cachedVoices.find(test);
    if (found) return found;
  }
  return cachedVoices[0] || null;
}

function speak(text, onDone) {
  if (!window.speechSynthesis) { onDone && onDone(); return; }
  window.speechSynthesis.cancel();

  // Wait a tick after cancel before speaking (Chrome stability fix)
  setTimeout(() => {
    const clean = text.replace(/[*_#`~\[\]()]/g, "").replace(/\s+/g, " ").trim();
    const utter = new SpeechSynthesisUtterance(clean);
    utter.voice = getBestVoice();
    utter.rate  = 1.0;
    utter.pitch = 1.0;

    utter.onstart = () => visualizerEl.classList.add("active");
    utter.onend   = () => {
      visualizerEl.classList.remove("active");
      onDone && onDone();
    };
    utter.onerror = () => {
      visualizerEl.classList.remove("active");
      onDone && onDone();
    };

    window.speechSynthesis.speak(utter);
  }, 100);
}

function toggleMic() {
  if (!recognition) { alert("Voice recognition is not supported in this browser."); return; }
  if (isListening) {
    recognition.stop();
  } else {
    window.speechSynthesis?.cancel();
    recognition.start();
  }
}

// ═══════════════════════════════════════════
// CHAT HELPERS
// ═══════════════════════════════════════════
function fillInput(btn) { inputEl.value = btn.textContent; send(); }
function hideEmpty()    { if (emptyState) emptyState.style.display = "none"; }
function scroll()       { messagesEl.scrollTop = messagesEl.scrollHeight; }
function esc(s)         { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function shortenUrl(url) {
  try {
    const slug = new URL(url).pathname.split("/").filter(Boolean).pop() || url;
    return slug.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  } catch { return url; }
}

function addUserMsg(text) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  messagesEl.appendChild(el);
  scroll();
}

function addBotMsg(text, sources=[], titles=[], modelUsed="", recommendations=[]) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg bot";

  let srcHtml = "";
  const shown = sources.slice(0, 3);
  if (shown.length) {
    const links = shown.map((url, i) => {
      const label = (titles && titles[i]) ? titles[i] : shortenUrl(url);
      return `<a class="source-link" href="${esc(url)}" target="_blank">${esc(label)}</a>`;
    }).join("");
    srcHtml = `<div class="sources"><div class="sources-title">Sources</div>${links}</div>`;
  }

  let recHtml = "";
  if (recommendations?.length) {
    const btns = recommendations.map(opt =>
      `<button class="opt-btn" data-action="send-option" data-value="${esc(opt)}">${esc(opt)}</button>`
    ).join("");
    recHtml = `<div class="recommendations">${btns}</div>`;
  }

  el.innerHTML = `
    <div class="bot-icon-row"><span class="bot-sparkle">✦</span></div>
    <div class="bubble">${marked.parse(text)}</div>
    ${srcHtml}${recHtml}`;
  messagesEl.appendChild(el);
  createIcons();
  scroll();
}

function addThinking() {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg bot thinking";
  el.id = "thinking";
  el.innerHTML = `
    <div class="bot-icon-row"><span class="bot-sparkle anim">✦</span></div>
    <div class="bubble"><div class="shimmer-dots"><span></span><span></span><span></span></div></div>`;
  messagesEl.appendChild(el);
  scroll();
}

function removeThinking() {
  const t = document.getElementById("thinking");
  if (t) t.remove();
}

function renderNoInfo(searchUrl) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `
    <div class="bot-icon-row"><span class="bot-sparkle">✦</span></div>
    <div class="no-info-card">
      <div class="no-info-icon">🔍</div>
      <div class="no-info-title">Not found in knowledge base</div>
      <div class="no-info-subtitle">Try rephrasing, or search the Hushly Help Center directly.</div>
      <a class="no-info-link" href="${esc(searchUrl)}" target="_blank">Search Help Center →</a>
    </div>`;
  messagesEl.appendChild(el);
  scroll();
}

function sendOption(opt) { inputEl.value = opt; send(); }

// ═══════════════════════════════════════════
// ENHANCE QUERY
// ═══════════════════════════════════════════
async function enhanceQuery() {
  const query = inputEl.value.trim();
  if (!query) return;
  enhanceBtn.disabled = true;
  try {
    const res = await fetch(`${BACKEND_URL}/enhance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history: chatHistory.slice(-4) })
    });
    if (res.ok) {
      const data = await res.json();
      inputEl.value = data.enhanced;
      inputEl.focus();
    }
  } catch (e) { console.warn("Enhance failed:", e); }
  finally { enhanceBtn.disabled = false; }
}

// ═══════════════════════════════════════════
// MAIN SEND
// ═══════════════════════════════════════════
async function send() {
  const query = inputEl.value.trim();
  if (!query || sendBtn.disabled) return;
  inputEl.value = "";
  addUserMsg(query);

  const q = query.toLowerCase();

  // ── Guide prompt intercept (waiting for voice/live choice via text) ──
  if (inGuidePrompt) {
    handleGuidePromptText(q, query);
    return;
  }

  // ── Voice guide active — intercept navigation commands ──
  if (guidedMode === "voice" && guidedData) {
    if (handleGuideNavCommand(q)) return;
    // Non-guide question: suspend guide and answer normally
    guidedMode = null;
    updateGuideModeBar();
    addBotMsg("*(Guide paused to answer your question)*");
  }

  // ── Live guide active — intercept navigation commands ──
  if (guidedMode === "live" && guidedData) {
    if (handleGuideNavCommand(q)) return;
    // Non-guide question: stop live guide
    stopGuide();
    addBotMsg("*(Guide stopped. Answering your question...)*");
  }

  addThinking();

  try {
    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history: chatHistory.slice(-4) })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Server error");
    }
    const data = await res.json();
    removeThinking();

    if (data.no_info) {
      const url = `https://hushly.freshdesk.com/support/search/solutions?term=${encodeURIComponent(query)}`;
      renderNoInfo(url);
      chatHistory.push({ role: "user", content: query });
      chatHistory.push({ role: "assistant", content: "[NO_INFO]" });
      return;
    }

    addBotMsg(data.answer, data.sources || [], data.titles || [], data.model_used || "", data.recommendations || []);
    chatHistory.push({ role: "user", content: query });
    chatHistory.push({ role: "assistant", content: data.answer });
    lastQuery  = query;
    lastAnswer = data.answer;

    // Silently check if answer has actionable steps
    checkActionable(data.answer);

  } catch (err) {
    removeThinking();
    addBotMsg(`⚠️ ${err.message}`);
  }
}

// (keypress listener registered in setupEventDelegation)

// ═══════════════════════════════════════════
// ACTIONABLE STEP DETECTION
// ═══════════════════════════════════════════
async function checkActionable(answerText) {
  try {
    const res = await fetch(`${BACKEND_URL}/generate_steps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer: answerText })
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.is_actionable_task && data.steps?.length) {
      guidedData = data;
      inGuidePrompt = true;
      showGuidePromptCard(data.task_title);
    }
  } catch (e) { console.warn("generate_steps error:", e); }
}

function showGuidePromptCard(taskTitle) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg bot";
  el.id = "guide-prompt-msg";
  el.innerHTML = `
    <div class="msg-label-row"><span class="msg-label">Guide Available</span></div>
    <div class="guide-prompt-card">
      <div class="guide-card-title">Interactive Guide</div>
      <div class="guide-card-task">How to: ${esc(taskTitle)}</div>
      <div class="guide-btns-row">
        <button class="guide-choice-btn voice-btn" data-action="choose-voice-guide">
          <span class="btn-icon">🎤</span>
          <span class="btn-label">Voice Guide</span>
          <span class="btn-desc">Spoken step-by-step</span>
        </button>
        <button class="guide-choice-btn live-btn" data-action="choose-live-guide">
          <span class="btn-icon">👆</span>
          <span class="btn-label">Live Guide</span>
          <span class="btn-desc">Visual highlights on page</span>
        </button>
      </div>
    </div>`;
  messagesEl.appendChild(el);
  createIcons();
  scroll();

  // If last input was voice, speak the offer
  if (lastInputMethod === "voice") {
    speak(`I can guide you through ${taskTitle} step by step. Say voice guide or live guide to choose.`);
  }
}

// ── Handle text response to guide prompt ──
function handleGuidePromptText(qLower, originalText) {
  if (qLower.match(/voice|speak|talk|audio|say/)) {
    inGuidePrompt = false;
    removeGuidePromptCard();
    chooseVoiceGuide();
  } else if (qLower.match(/live|visual|highlight|show|click|screen/)) {
    inGuidePrompt = false;
    removeGuidePromptCard();
    chooseLiveGuide();
  } else if (qLower.match(/no|nope|nah|skip|cancel|dismiss/)) {
    inGuidePrompt = false;
    removeGuidePromptCard();
    addBotMsg("Got it! Let me know if you need help with anything else.");
  } else {
    // Unclear — ask again
    addBotMsg('Please choose: **Voice Guide** (spoken steps) or **Live Guide** (visual highlights on the Hushly app). Or say "skip" to dismiss.');
  }
}

// ── Handle voice response to guide prompt ──
function handleGuidePromptVoice(transcript) {
  const t = transcript.toLowerCase();
  handleGuidePromptText(t, transcript);
}

function removeGuidePromptCard() {
  const el = document.getElementById("guide-prompt-msg");
  if (el) el.remove();
}

// ═══════════════════════════════════════════
// VOICE GUIDE
// ═══════════════════════════════════════════
function chooseVoiceGuide() {
  addUserMsg("Start Voice Guide");
  startVoiceGuide();
}

function startVoiceGuide() {
  guidedMode = "voice";
  currentStep = 0;
  updateGuideModeBar();
  addBotMsg(`**Voice Guide started: ${guidedData.task_title}**\n\nSay **"next"** / **"I clicked"** to advance, **"back"** to go back, **"repeat"** to replay, or **"stop"** to exit.`);
  readCurrentVoiceStep();
  // Start listening automatically
  if (recognition && !isListening) {
    setTimeout(() => { try { recognition.start(); } catch {} }, 600);
  }
}

function readCurrentVoiceStep() {
  const step = guidedData.steps[currentStep];
  const text = step.text || step;
  const total = guidedData.steps.length;

  // Add the step card to chat
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `
    <div class="msg-label-row"><span class="msg-label">Voice Guide</span></div>
    <div class="voice-step-card">
      <div class="step-num">Step ${currentStep + 1} of ${total}</div>
      <div class="step-text">${esc(text)}</div>
      <div class="step-hint">🎤 Say "next" or "I clicked" to continue</div>
    </div>`;
  messagesEl.appendChild(el);
  scroll();

  // Update bar
  guideStepCounter.textContent = `Step ${currentStep + 1} of ${total}`;

  // Speak the step
  speak(`Step ${currentStep + 1}: ${text}`, () => {
    // After speaking, re-activate mic for voice guide
    if (guidedMode === "voice" && recognition && !isListening) {
      try { recognition.start(); } catch {}
    }
  });
}

function handleVoiceGuideCommand(transcript) {
  const t = transcript.toLowerCase();
  if (t.match(/\b(next|done|continue|i clicked|finished|ok|okay|yes|yep|got it|moved on)\b/)) {
    nextStep();
  } else if (t.match(/\b(back|previous|go back)\b/)) {
    prevStep();
  } else if (t.match(/\b(repeat|again|say that again|what was that)\b/)) {
    addBotMsg("*(Repeating step)*");
    readCurrentVoiceStep();
  } else if (t.match(/\b(stop|cancel|quit|exit|end guide)\b/)) {
    stopGuide();
  } else {
    // Unknown command during guide — echo and re-prompt
    addBotMsg(`*(Heard: "${esc(transcript)}")*\n\nSay **"next"**, **"back"**, **"repeat"**, or **"stop"**.`);
    speak("Say next to continue, back to go back, or stop to exit.", () => {
      if (recognition && !isListening) try { recognition.start(); } catch {}
    });
  }
}

// ═══════════════════════════════════════════
// LIVE GUIDE — smart path-memory flow
// ═══════════════════════════════════════════
function chooseLiveGuide() {
  addUserMsg("Start Live Guide");
  startSmartLiveGuide();
}

async function startSmartLiveGuide() {
  // ── Step 1: check path store (no scan needed) ──
  addThinking();
  try {
    const checkRes = await fetch(`${BACKEND_URL}/smart_guide`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: lastQuery, answer: lastAnswer })
    });
    if (!checkRes.ok) throw new Error("smart_guide failed");
    const checkData = await checkRes.json();
    removeThinking();

    if (checkData.needs_scan) {
      // ── Step 2: no cached path — scan page elements ──
      await scanAndGuide();
    } else {
      // ── Cached path found ──
      guidedData    = checkData;
      currentPathId = checkData.id || null;
      const badge   = checkData.from_cache
        ? "✓ **Saved guide path found** — no page scan needed."
        : "AI-generated guide.";
      startLiveGuide(badge);
    }
  } catch (e) {
    removeThinking();
    addBotMsg(`⚠️ Could not start guide: ${e.message}`);
  }
}

async function scanAndGuide() {
  addBotMsg("*Scanning page elements…*");
  try {
    const tab = await getActiveTab();
    if (!tab) { addBotMsg("⚠️ No active tab found."); return; }

    await ensureContentScript(tab.id);

    const scanRes = await chrome.tabs.sendMessage(tab.id, { type: "HUSHLY_SCAN_PAGE" })
      .catch(err => { console.error("SCAN_PAGE failed:", err); return null; });
    const elements = scanRes?.elements || [];

    console.log("[Hushly] Page scan found", elements.length, "elements:", elements.slice(0, 10));

    if (!elements.length) {
      addBotMsg("⚠️ No interactive elements found. Make sure the Hushly app is open and the page is fully loaded, then try again.");
      return;
    }

    addThinking();
    const guideRes = await fetch(`${BACKEND_URL}/smart_guide`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query:         lastQuery,
        answer:        lastAnswer,
        page_elements: elements
      })
    });
    removeThinking();
    if (!guideRes.ok) throw new Error("smart_guide LLM failed");

    const guideData = await guideRes.json();

    // Enrich steps with selectors from scan (LLM picks element_idx, we resolve selector)
    if (guideData.steps) {
      guideData.steps = guideData.steps.map(step => {
        if (step.element_idx != null && step.element_idx >= 0) {
          const scanned = elements.find(e => e.idx === step.element_idx);
          if (scanned) {
            step.selector          = scanned.selector;
            step.selector_fallbacks = [];
            step.element_text      = step.element_text || scanned.text;
            step.element_hint      = step.element_hint || scanned.text;
          }
        }
        return step;
      });
    }

    guidedData    = guideData;
    currentPathId = null; // not yet saved
    startLiveGuide("AI-generated guide — path will be saved after you complete it.");

  } catch (e) {
    removeThinking();
    addBotMsg(`⚠️ Scan failed: ${e.message}`);
  }
}

async function startLiveGuide(statusNote) {
  guidedMode  = "live";
  currentStep = 0;
  updateGuideModeBar();

  // ── Auto-hide the chat UI so user sees the full page ──
  if (IS_SIDEBAR) {
    // Hide the sidebar panel — iframe JS keeps running, port stays alive
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab) chrome.tabs.sendMessage(tab.id, { type: "HUSHLY_HIDE_SIDEBAR" }).catch(() => {});
    } catch {}
  } else {
    // Collapse popup to mini status bar
    document.body.classList.add("guide-active");
    document.getElementById("gmbTitle").textContent = guidedData.task_title || "Guide running";
    document.getElementById("gmbStep").textContent  = `Step 1 of ${guidedData.steps?.length || "?"}`;
  }

  await connectContentPort();
  saveGuideSession();
  sendCurrentLiveStep();
}

// ── Tab / content-script helpers ────────────────────────────────────────────
async function getActiveTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    return tab || null;
  } catch { return null; }
}

async function ensureContentScript(tabId) {
  try {
    const pong = await chrome.tabs.sendMessage(tabId, { type: "HUSHLY_PING" }).catch(() => null);
    if (!pong) {
      // Content script not running yet — inject it
      await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
      await chrome.scripting.insertCSS({ target: { tabId }, files: ["guide-overlay.css"] });
      await new Promise(r => setTimeout(r, 400));
    }
  } catch (e) {
    console.warn("ensureContentScript:", e);
  }
}

async function connectContentPort() {
  if (typeof chrome === "undefined" || !chrome.tabs) return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return;
    contentPort = chrome.tabs.connect(tab.id, { name: "hushly-guide-port" });
    contentPort.onMessage.addListener(handleContentPortMessage);
    contentPort.onDisconnect.addListener(() => handlePortDisconnect());
  } catch (e) {
    console.warn("Could not connect to content script:", e);
    addBotMsg("⚠️ Could not connect to the page. Make sure you have the Hushly app open in a tab, then try again.");
    guidedMode = null;
    updateGuideModeBar();
  }
}

async function handlePortDisconnect() {
  contentPort = null;
  if (guidedMode !== "live" || !guidedData) return;

  // Page navigated — wait for new page to finish loading, then reconnect
  addBotMsg("*(Page navigating — reconnecting guide in a moment…)*");
  await new Promise(r => setTimeout(r, 1800));

  try {
    const tab = await getActiveTab();
    if (!tab) throw new Error("no tab");
    await ensureContentScript(tab.id);
    await connectContentPort();
    // Re-send the current step so the new page gets the highlight
    if (currentStep < guidedData.steps.length) {
      await new Promise(r => setTimeout(r, 300));
      sendCurrentLiveStep();
    }
  } catch (e) {
    addBotMsg("*(Could not reconnect automatically. Click **Done / Next →** in the step card above to continue.)*");
  }
}

function sendCurrentLiveStep() {
  if (!guidedData || currentStep >= guidedData.steps.length) return;

  const step  = guidedData.steps[currentStep];
  const total = guidedData.steps.length;
  const msg   = {
    type:       "HUSHLY_GUIDE_STEP",
    step: {
      text:               step.text || step,
      element_hint:       step.element_hint || step.element_text || "",
      selector:           step.selector || "",
      selector_fallbacks: step.selector_fallbacks || [],
      from_cache:         guidedData.from_cache || false
    },
    stepNum:    currentStep + 1,
    totalSteps: total,
    taskTitle:  guidedData.task_title,
    pathId:     currentPathId || guidedData.id || ""
  };

  if (contentPort) {
    try { contentPort.postMessage(msg); } catch {}
  }

  showLiveStepCard(step, currentStep + 1, total);
}

function showLiveStepCard(step, num, total) {
  const text = step.text || step;
  const hint = step.element_hint || "";
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `
    <div class="msg-label-row"><span class="msg-label">Live Guide</span></div>
    <div class="live-step-card">
      <div class="step-num">Step ${num} of ${total}</div>
      <div class="step-text">${esc(text)}</div>
      ${hint ? `<div class="step-hint">👆 Look for: <strong>${esc(hint)}</strong></div>` : ""}
      <div class="step-actions">
        <button class="step-action-btn secondary" data-action="prev-step">← Back</button>
        <button class="step-action-btn primary"   data-action="next-step">Done / Next →</button>
      </div>
    </div>`;
  messagesEl.appendChild(el);
  guideStepCounter.textContent = `Step ${num} of ${total}`;
  scroll();
}

function handleContentPortMessage(msg) {
  switch (msg.type) {
    case "STEP_COMPLETED":   nextStep();   break;
    case "STEP_BACK":        prevStep();   break;
    case "GUIDE_STOP_REQUESTED": stopGuide(); break;
    case "ELEMENT_NOT_FOUND": break; // dock already shows the status
    case "STEP_FEEDBACK":
      logStepFeedback(msg.path_id, msg.step_idx, msg.issue, msg.page_url);
      break;
  }
}

async function logStepFeedback(pathId, stepIdx, issue, pageUrl) {
  try {
    await fetch(`${BACKEND_URL}/guide/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path_id:  pathId,
        step_idx: stepIdx,
        issue,
        page_url: pageUrl,
        comment:  ""
      })
    });
    addBotMsg("*(Feedback logged — admin will review and update this guide path.)*");
  } catch {}
}

// ═══════════════════════════════════════════
// SHARED GUIDE NAVIGATION
// ═══════════════════════════════════════════
function handleGuideNavCommand(qLower) {
  if (qLower.match(/\b(next|done|continue|i clicked|finished|ok|okay|yes|yep|got it)\b/)) {
    nextStep(); return true;
  }
  if (qLower.match(/\b(back|previous|go back)\b/)) {
    prevStep(); return true;
  }
  if (qLower.match(/\b(repeat|again)\b/)) {
    if (guidedMode === "voice") readCurrentVoiceStep();
    else if (guidedMode === "live") { sendCurrentLiveStep(); }
    return true;
  }
  if (qLower.match(/\b(stop|cancel|quit|exit)\b/)) {
    stopGuide(); return true;
  }
  return false;
}

function nextStep() {
  currentStep++;
  if (currentStep >= guidedData.steps.length) {
    completeGuide();
  } else {
    if (guidedMode === "voice") readCurrentVoiceStep();
    else if (guidedMode === "live") {
      sendCurrentLiveStep();
      saveGuideSession();
      updateMiniBar();
    }
  }
}

function updateMiniBar() {
  const stepEl = document.getElementById("gmbStep");
  if (stepEl && guidedData) {
    stepEl.textContent = `Step ${currentStep + 1} of ${guidedData.steps.length}`;
  }
}

function prevStep() {
  if (currentStep > 0) currentStep--;
  if (guidedMode === "voice") {
    addBotMsg("*(Going back one step)*");
    readCurrentVoiceStep();
  } else if (guidedMode === "live") {
    sendCurrentLiveStep();
  }
}

function completeGuide() {
  const mode = guidedMode;
  guidedMode = null;
  updateGuideModeBar();

  if (contentPort) {
    try { contentPort.postMessage({ type: "HUSHLY_GUIDE_STOP" }); } catch {}
    contentPort = null;
  }

  clearGuideSession();
  document.body.classList.remove("guide-active");  // restore popup

  // Auto-save path if it was freshly AI-generated (not from cache)
  if (guidedData && !guidedData.from_cache && guidedData.steps?.length) {
    saveGuidePath(guidedData);
  }

  addBotMsg("🎉 **All done! Task completed successfully.**\n\nWhat else can I help you with?");
  if (mode === "voice") speak("Great job! Task complete. What else can I help you with?");
}

async function saveGuidePath(data) {
  try {
    const res = await fetch(`${BACKEND_URL}/guide/save_path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_title: data.task_title,
        keywords:   data.keywords || [],
        steps:      data.steps
      })
    });
    if (res.ok) {
      const saved = await res.json();
      currentPathId = saved.saved;
    }
  } catch {}
}

function stopGuide() {
  const wasVoice = guidedMode === "voice";
  guidedMode = null;
  inGuidePrompt = false;
  updateGuideModeBar();

  // Stop speech & recognition
  window.speechSynthesis?.cancel();
  if (isListening) { try { recognition?.stop(); } catch {} }

  // Tell content script to remove overlay
  if (contentPort) {
    try { contentPort.postMessage({ type: "HUSHLY_GUIDE_STOP" }); } catch {}
    contentPort = null;
  }

  clearGuideSession();
  document.body.classList.remove("guide-active");  // restore popup
  addBotMsg("*(Guide stopped. Ask another question whenever you're ready.)*");
}

// ═══════════════════════════════════════════
// PAGE CONTEXT ANALYSIS
// ═══════════════════════════════════════════
async function analyzePage() {
  if (typeof chrome === "undefined") return;

  hideEmpty();
  addUserMsg("📄 Explain this page");
  addThinking();

  try {
    // Get current tab info
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) {
      removeThinking();
      addBotMsg("⚠️ Could not access current tab.");
      return;
    }

    // Ensure content script is injected
    await ensureContentScript(tab.id);

    // Scan page elements
    const pageData = await chrome.tabs.sendMessage(tab.id, { type: "HUSHLY_SCAN_PAGE" });

    // Build context payload
    const payload = {
      url: tab.url,
      title: tab.title,
      page_elements: pageData?.elements?.slice(0, 50) || [],  // Top 50 elements
      query: "What is this page about and how do I use it?"
    };

    // Send to backend
    const res = await fetch(`${BACKEND_URL}/analyze_page`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    removeThinking();

    if (!res.ok) {
      const err = await res.json();
      addBotMsg(`⚠️ ${err.error || "Could not analyze page"}`);
      return;
    }

    const data = await res.json();

    if (data.no_info) {
      addBotMsg("I don't recognize this page in the knowledge base. Try asking a specific question about what you're trying to do.");
    } else {
      // Render structured page analysis
      renderPageAnalysis(data);
      chatHistory.push({ role: "user", content: "Explain this page" });
      chatHistory.push({ role: "assistant", content: data.page_context || "" });
    }

  } catch (e) {
    removeThinking();
    console.warn("analyzePage error:", e);
    addBotMsg("⚠️ Could not analyze page. Make sure you're on a Hushly page and the server is running.");
  }
}

// Render structured page analysis with sections
function renderPageAnalysis(data) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg bot page-analysis";

  // Build sections
  let html = `
    <div class="bot-icon-row"><span class="bot-sparkle">✦</span></div>
    <div class="bubble">
      <div class="page-analysis-card">
        <div class="page-analysis-header">
          <span class="page-icon">📄</span>
          <span class="page-title">Page Analysis</span>
        </div>
  `;

  // Page Context section (2-5 lines)
  if (data.page_context) {
    html += `
        <div class="analysis-section">
          <div class="section-title">Page Context</div>
          <div class="section-content context-text">${esc(data.page_context)}</div>
        </div>
    `;
  }

  // Key Features section (bullet points)
  if (data.key_features && data.key_features.length > 0) {
    html += `
        <div class="analysis-section">
          <div class="section-title">Key Features</div>
          <ul class="feature-list">
    `;
    data.key_features.forEach(feature => {
      html += `<li>${esc(feature)}</li>`;
    });
    html += `</ul></div>`;
  }

  // Navigation Summary section
  if (data.navigation_summary && data.navigation_summary.length > 0) {
    html += `
        <div class="analysis-section">
          <div class="section-title">Navigation Summary</div>
          <div class="nav-summary-list">
    `;
    data.navigation_summary.forEach(item => {
      html += `
        <div class="nav-item">
          <span class="nav-section">${esc(item.section)}</span>
          <span class="nav-arrow">→</span>
          <span class="nav-purpose">${esc(item.purpose)}</span>
        </div>
      `;
    });
    html += `</div></div>`;
  }

  // Sources
  if (data.sources && data.sources.length > 0) {
    const shown = data.sources.slice(0, 3);
    const links = shown.map((url, i) => {
      const label = (data.titles && data.titles[i]) ? data.titles[i] : shortenUrl(url);
      return `<a class="source-link" href="${esc(url)}" target="_blank">${esc(label)}</a>`;
    }).join("");
    html += `<div class="sources"><div class="sources-title">Sources</div>${links}</div>`;
  }

  html += `</div></div>`;
  el.innerHTML = html;
  messagesEl.appendChild(el);
  createIcons();
  scroll();
}

// Helper to ensure content script is injected
async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "HUSHLY_PING" });
  } catch {
    // Not injected yet, inject it
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content.js"]
    });
    await new Promise(r => setTimeout(r, 100));
  }
}

function updateGuideModeBar() {
  if (!guidedMode) {
    guideModeBar.style.display = "none";
    return;
  }
  guideModeBar.style.display = "flex";
  if (guidedMode === "voice") {
    guideModeIcon.textContent = "🎤";
    guideModeLabel.textContent = "Voice Guide Active";
  } else {
    guideModeIcon.textContent = "👆";
    guideModeLabel.textContent = "Live Guide Active";
  }
  const total = guidedData?.steps?.length || 0;
  guideStepCounter.textContent = total ? `Step ${currentStep + 1} of ${total}` : "";
}

// ═══════════════════════════════════════════
// EXPOSE FUNCTIONS TO HTML onclick
// All interactions handled by setupEventDelegation() — no window.* exports needed.

// ═══════════════════════════════════════════
// POP-OUT WINDOW
// ═══════════════════════════════════════════
async function popOut() {
  if (typeof chrome === "undefined" || !chrome.windows) return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.storage.session.set({
      popoutTargetTabId: tab?.id ?? null,
      popoutChatHistory: chatHistory.slice(-30),
      popoutBackendUrl:  BACKEND_URL
    });
    await chrome.windows.create({
      url:     chrome.runtime.getURL("popup.html") + "?popout=1",
      type:    "popup",
      width:   440,
      height:  660,
      focused: true
    });
    window.close();
  } catch (e) {
    console.warn("popOut:", e);
  }
}

async function restorePopoutHistory() {
  try {
    const data = await chrome.storage.session.get([
      "popoutChatHistory", "popoutTargetTabId", "popoutBackendUrl"
    ]);
    if (data.popoutBackendUrl) {
      BACKEND_URL = data.popoutBackendUrl;
      const inp = document.getElementById("backendUrlInput");
      if (inp) inp.value = BACKEND_URL;
    }
    if (data.popoutTargetTabId) {
      // Store for later use by attachToPage
      window.__popoutTargetTabId = data.popoutTargetTabId;
    }
    if (data.popoutChatHistory?.length) {
      chatHistory = data.popoutChatHistory;
      hideEmpty();
      // Replay messages into the chat view
      for (const msg of chatHistory) {
        if (msg.role === "user") {
          const el = document.createElement("div");
          el.className = "msg user";
          el.innerHTML = `<div class="bubble">${esc(msg.content)}</div>`;
          messagesEl.appendChild(el);
        } else if (msg.role === "assistant" && msg.content !== "[NO_INFO]") {
          const el = document.createElement("div");
          el.className = "msg bot";
          el.innerHTML = `<div class="bot-icon-row"><span class="bot-sparkle">✦</span></div>
            <div class="bubble">${marked.parse(msg.content)}</div>`;
          messagesEl.appendChild(el);
        }
      }
      scroll();
    }
  } catch (e) {
    console.warn("restorePopoutHistory:", e);
  }
}

async function attachToPage() {
  if (typeof chrome === "undefined") return;
  try {
    // Try the original tab first, then any non-chrome tab
    let tabId = window.__popoutTargetTabId;
    if (!tabId) {
      const tabs = await chrome.tabs.query({});
      const tab  = tabs.find(t =>
        t.url && !t.url.startsWith("chrome://") && !t.url.startsWith("chrome-extension://")
      );
      tabId = tab?.id;
    }
    if (!tabId) { addBotMsg("⚠️ No web page found to attach to."); return; }

    // Save history so the sidebar can pick it up
    await chrome.storage.session.set({ popoutChatHistory: chatHistory.slice(-30) });

    await ensureContentScript(tabId);
    await chrome.tabs.sendMessage(tabId, { type: "HUSHLY_SHOW_SIDEBAR" }).catch(() => {});
    window.close();
  } catch (e) {
    addBotMsg("⚠️ Could not attach: " + e.message);
  }
}

// Make the chrome popup window draggable via the drag handle
function initPopoutDrag() {
  if (typeof chrome === "undefined" || !chrome.windows) return;
  const handle = document.getElementById("popoutDragHandle");
  if (!handle) return;

  let dragging = false, startX = 0, startY = 0, winLeft = 0, winTop = 0;

  handle.addEventListener("mousedown", async (e) => {
    e.preventDefault();
    dragging = true;
    startX = e.screenX;
    startY = e.screenY;
    const [win] = await chrome.windows.getAll({ populate: false })
      .then(wins => wins.filter(w => w.focused));
    if (win) { winLeft = win.left; winTop = win.top; }
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.screenX - startX;
    const dy = e.screenY - startY;
    chrome.windows.getCurrent().then(win => {
      chrome.windows.update(win.id, { left: winLeft + dx, top: winTop + dy });
    }).catch(() => {});
  });

  document.addEventListener("mouseup", () => { dragging = false; });
}

// ═══════════════════════════════════════════
// SAFE ICON RENDERER
// ═══════════════════════════════════════════
function createIcons() {
  if (typeof lucide !== "undefined") {
    try { lucide.createIcons(); } catch (e) { console.warn("lucide error:", e); }
  }
}

// ═══════════════════════════════════════════
// EVENT DELEGATION
// Handles ALL button/interactive clicks in the popup.
// Replaces every onclick="..." attribute (blocked by MV3 CSP).
// ═══════════════════════════════════════════
function setupEventDelegation() {
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;

    const action = btn.dataset.action;
    switch (action) {

      // ── Static header / input buttons ──
      case "open-page-assistant": openPageAssistant(); break;
      case "toggle-settings":     toggleSettings();    break;
      case "save-settings":       saveSettings();      break;
      case "stop-guide":          stopGuide();         break;
      case "enhance-query":       enhanceQuery();      break;
      case "toggle-mic":          toggleMic();         break;
      case "send":                send();              break;
      case "pop-out":             popOut();            break;
      case "attach-to-page":      attachToPage();      break;
      case "analyze-page":        analyzePage();       break;

      // ── Suggestion chips (fill input with chip text) ──
      case "fill-input":
        inputEl.value = btn.textContent.trim();
        send();
        break;

      // ── Recommendation option buttons ──
      case "send-option":
        sendOption(btn.dataset.value || btn.textContent.trim());
        break;

      // ── Guide choice card ──
      case "choose-voice-guide": chooseVoiceGuide(); break;
      case "choose-live-guide":  chooseLiveGuide();  break;

      // ── Live guide step navigation ──
      case "prev-step": prevStep(); break;
      case "next-step": nextStep(); break;
    }
  });

  // Enter key sends message
  inputEl.addEventListener("keypress", (e) => { if (e.key === "Enter") send(); });
}

// ═══════════════════════════════════════════
// BOOTSTRAP
// ═══════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  createIcons();           // render <i data-lucide="..."> icons
  setupEventDelegation();  // wire all button clicks (no inline onclick needed)
  init();                  // load settings, server health-check, voice init
});
