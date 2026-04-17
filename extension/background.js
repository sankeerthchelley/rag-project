/* ═══════════════════════════════════════════
   Hushly Guide Extension — Service Worker
   ═══════════════════════════════════════════ */

// Keep service worker alive during active guide sessions
// (Chrome MV3 service workers can be terminated between events)

chrome.runtime.onInstalled.addListener(() => {
  console.log("Hushly Guide Assistant installed.");
});

// Relay messages that content scripts might send without a port open
// (e.g., if popup was closed mid-guide and content sends a completion)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "STEP_COMPLETED" || message.type === "ELEMENT_NOT_FOUND") {
    // Forward to any open extension pages (popup, if open)
    chrome.runtime.sendMessage(message).catch(() => {
      // Popup not open — ignore
    });
  }
  return false;
});

// When a tab navigates (user clicks a link during live guide),
// notify the popup so it can handle reconnection.
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    // Notify popup that the page reloaded — it will attempt re-injection if needed
    chrome.runtime.sendMessage({ type: "TAB_NAVIGATED", tabId, url: tab.url }).catch(() => {});
  }
});
