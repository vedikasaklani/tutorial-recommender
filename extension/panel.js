const DEFAULT_API_URL = "http://localhost:5000/recommend";

const backendInput = document.getElementById("backend");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

chrome.storage.local.get("apiUrl", ({ apiUrl }) => {
  backendInput.value = apiUrl || DEFAULT_API_URL;
});

backendInput.addEventListener("change", () => {
  chrome.storage.local.set({ apiUrl: backendInput.value.trim() });
});

document.getElementById("go").addEventListener("click", async () => {
  resultsEl.innerHTML = "";
  const apiUrl = backendInput.value.trim() || DEFAULT_API_URL;

  statusEl.textContent = "Reading current tab...";
  let tab;
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  } catch (err) {
    statusEl.textContent = `Could not read the current tab: ${err.message}`;
    return;
  }
  if (!tab?.url || !tab.url.toLowerCase().includes(".pdf")) {
    statusEl.textContent = "Open a PDF tab first.";
    return;
  }

  statusEl.textContent = "Fetching the PDF...";
  let pdfBlob;
  try {
    const pdfResp = await fetch(tab.url);
    if (!pdfResp.ok) throw new Error(`server returned HTTP ${pdfResp.status}`);
    pdfBlob = await pdfResp.blob();
  } catch (err) {
    statusEl.textContent =
      `Could not fetch the PDF from ${tab.url}: ${err.message}\n` +
      `If this is a local file:// PDF, enable "Allow access to file URLs" for this extension in chrome://extensions.`;
    return;
  }

  statusEl.textContent = "Generating recommendations... this can take a minute.";
  let videos;
  try {
    const form = new FormData();
    form.append("pdf", pdfBlob, "document.pdf");
    const resp = await fetch(apiUrl, { method: "POST", body: form });
    const body = await resp.json().catch(() => null);
    if (!resp.ok || body?.error) {
      throw new Error(body?.error || `backend returned HTTP ${resp.status}`);
    }
    videos = body;
  } catch (err) {
    statusEl.textContent = `Could not get recommendations from ${apiUrl}: ${err.message}`;
    return;
  }

  statusEl.textContent = videos.length ? "" : "No recommendations found.";
  for (const v of videos) {
    const div = document.createElement("div");
    div.className = "rec";

    const link = document.createElement("a");
    link.href = v.link;
    link.target = "_blank";
    link.textContent = v.title;

    const desc = document.createElement("p");
    desc.textContent = v.description;

    div.append(link, desc);
    resultsEl.appendChild(div);
  }
});
