// ===== STATE =====
let state = {
  page: "dashboard",
  analyzed: false,
  analyzing: false,
  lastAnalysis: null,
  selectedImages: [],
  chatMessages: [
    {
      role: "ai",
      text: "مددگار here! 🎯\n\nAsk me anything about your opportunities, deadlines, or applications.",
    },
  ],
};

const API_BASE = "";
const MAX_IMAGES = 20;

const pages = [
  { id: "dashboard", label: "Dashboard" },
  { id: "analyzer", label: "Analyzer" },
  { id: "chat", label: "Chat" },
  { id: "profile", label: "Profile" },
];

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function getPriorityClass(score) {
  if (score >= 8) return "priority-high";
  if (score >= 5) return "priority-med";
  return "priority-low";
}

function switchPage(page) {
  state.page = page;

  document.querySelectorAll(".page").forEach((p) => {
    p.style.display = "none";
  });

  const activePage = document.getElementById(page);
  if (activePage) activePage.style.display = "block";

  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.classList.remove("active");
  });

  const activeTab = document.getElementById(`tab-${page}`);
  if (activeTab) activeTab.classList.add("active");
}

function buildNav() {
  const navTabs = document.getElementById("navTabs");
  navTabs.innerHTML = "";

  pages.forEach((page) => {
    const btn = document.createElement("button");
    btn.className = "nav-tab";
    btn.id = `tab-${page.id}`;
    btn.textContent = page.label;
    btn.onclick = () => switchPage(page.id);
    navTabs.appendChild(btn);
  });
}

function buildApp() {
  const app = document.getElementById("app");

  app.innerHTML = `
    <section id="dashboard" class="page">
      <div class="card mb-2">
        <div class="section-title">Opportunity Dashboard</div>
        <div class="section-sub">See the strongest opportunities ranked from your submitted text and images.</div>

        <div id="dashboardMetrics" class="grid-4 mb-2">
          <div class="metric-card">
            <div class="metric-val" id="metricTotal">0</div>
            <div class="metric-label">Ranked</div>
          </div>
          <div class="metric-card">
            <div class="metric-val" id="metricAnalyzed">0</div>
            <div class="metric-label">Inputs</div>
          </div>
          <div class="metric-card">
            <div class="metric-val" id="metricEdu">0</div>
            <div class="metric-label">EDU Detected</div>
          </div>
          <div class="metric-card">
            <div class="metric-val" id="metricProfile">No</div>
            <div class="metric-label">Profile Loaded</div>
          </div>
        </div>

        <div id="dashboardContent" class="grid-2"></div>
      </div>
    </section>

    <section id="analyzer" class="page" style="display:none;">
      <div class="grid-2">
        <div class="card">
          <div class="section-title">Analyze Emails / Images</div>
          <div class="section-sub">Paste text, attach up to 20 images, or drag and drop them here.</div>

          <label class="small-label" for="emailText">Paste text</label>
          <textarea
            id="emailText"
            class="input"
            rows="10"
            placeholder="Paste scholarship, internship, workshop, or admission emails here. Separate multiple entries with a line that says ---"
          ></textarea>

          <div class="mt-2">
            <label class="small-label">Attach images</label>

            <div id="dropZone" class="drop-zone" tabindex="0">
              <div class="drop-zone-title">Drag & drop images here</div>
              <div class="drop-zone-sub">or use the button below</div>

              <div class="mt-2">
                <input
                  id="imageInput"
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff,image/webp"
                  multiple
                  hidden
                />
                <button type="button" class="btn btn-ghost" onclick="openImagePicker()">Attach Images</button>
              </div>

              <div class="mt-2 small-note">Maximum 20 images</div>
            </div>
          </div>

          <div id="selectedImagesInfo" class="mt-2"></div>
          <div id="selectedImagesList" class="selected-files mt-1"></div>

          <div class="mt-2 flex gap-1 analyzer-actions">
            <button class="btn btn-primary" id="analyzeBtn" onclick="analyzeEmails()">Analyze</button>
            <button class="btn btn-ghost" onclick="loadSampleEmails()">Load Sample</button>
            <button class="btn btn-ghost" onclick="clearSelectedImages()">Clear Images</button>
          </div>

          <div id="analyzerStatus" class="mt-2"></div>
        </div>

        <div class="card">
          <div class="section-title">EDU Detection Snapshot</div>
          <div class="section-sub">Quick result from the backend classifier.</div>
          <div id="eduPredictions"></div>
        </div>
      </div>
    </section>

    <section id="chat" class="page chat-page" style="display:none;">
      <div class="chat-messages" id="chatBox"></div>
      <div class="chat-input-row">
        <input
          id="chatInput"
          class="input"
          type="text"
          placeholder="Ask about scholarships, deadlines, applications, or career guidance"
          onkeydown="handleChatKey(event)"
        />
        <button class="btn btn-primary" onclick="sendMessage()">Send</button>
      </div>
    </section>

    <section id="profile" class="page" style="display:none;">
      <div class="card">
        <div class="section-title">Student Profile</div>
        <div class="section-sub">This gets saved to Flask and used for personalized opportunity ranking.</div>

        <div class="grid-2">
          <div>
            <label class="small-label">Name</label>
            <input id="name" class="input" placeholder="Ali" />
          </div>
          <div>
            <label class="small-label">University</label>
            <input id="university" class="input" placeholder="NUST" />
          </div>
          <div>
            <label class="small-label">Degree</label>
            <input id="degree" class="input" placeholder="BSCS" />
          </div>
          <div>
            <label class="small-label">Semester</label>
            <input id="semester" class="input" placeholder="5" />
          </div>
          <div>
            <label class="small-label">GPA</label>
            <input id="gpa" class="input" placeholder="3.5" />
          </div>
          <div>
            <label class="small-label">Financial Need</label>
            <select id="financial_need" class="input">
              <option value="">Select</option>
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
          </div>
        </div>

        <div class="mt-2">
          <label class="small-label">Preferences</label>
          <input id="preferences" class="input" placeholder="AI, internships, scholarships" />
        </div>

        <div class="mt-2">
          <label class="small-label">Past Experience</label>
          <textarea id="past_experience" class="input" rows="4" placeholder="Python, ML projects, societies"></textarea>
        </div>

        <div class="mt-2 flex gap-1">
          <button class="btn btn-primary" onclick="saveProfile()">Save Profile</button>
        </div>

        <div id="profileStatus" class="mt-2"></div>
      </div>
    </section>
  `;
}

function initializeAnalyzerUpload() {
  const imageInput = document.getElementById("imageInput");
  const dropZone = document.getElementById("dropZone");

  if (!imageInput || !dropZone) return;

  imageInput.addEventListener("change", (event) => {
    handleSelectedFiles(event.target.files);
    imageInput.value = "";
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "dragend", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (files?.length) {
      handleSelectedFiles(files);
    }
  });

  dropZone.addEventListener("click", (event) => {
    if (event.target.tagName !== "BUTTON") {
      openImagePicker();
    }
  });

  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openImagePicker();
    }
  });

  renderSelectedImages();
}

function openImagePicker() {
  document.getElementById("imageInput")?.click();
}

function handleSelectedFiles(fileList) {
  const incoming = Array.from(fileList || []);
  const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff", "image/webp"];

  const filtered = incoming.filter(
    (file) => validTypes.includes(file.type) || /\.(png|jpe?g|bmp|tiff|webp)$/i.test(file.name)
  );

  if (!filtered.length) {
    setAnalyzerStatus("❌ Please choose valid image files only.");
    return;
  }

  const currentCount = state.selectedImages.length;
  const remainingSlots = MAX_IMAGES - currentCount;

  if (remainingSlots <= 0) {
    setAnalyzerStatus(`❌ You can upload a maximum of ${MAX_IMAGES} images.`);
    return;
  }

  const toAdd = filtered.slice(0, remainingSlots);
  state.selectedImages = [...state.selectedImages, ...toAdd];

  if (filtered.length > remainingSlots) {
    setAnalyzerStatus(`⚠️ Only the first ${MAX_IMAGES} images are allowed.`);
  } else {
    setAnalyzerStatus(`✅ ${toAdd.length} image(s) added.`);
  }

  renderSelectedImages();
}

function removeSelectedImage(index) {
  state.selectedImages.splice(index, 1);
  renderSelectedImages();
}

function clearSelectedImages() {
  state.selectedImages = [];
  renderSelectedImages();
  setAnalyzerStatus("Images cleared.");
}

function renderSelectedImages() {
  const info = document.getElementById("selectedImagesInfo");
  const list = document.getElementById("selectedImagesList");
  if (!info || !list) return;

  const count = state.selectedImages.length;
  info.innerHTML = `<span class="inline-status">${count} / ${MAX_IMAGES} images selected</span>`;

  if (!count) {
    list.innerHTML = "";
    return;
  }

  list.innerHTML = state.selectedImages
    .map(
      (file, index) => `
        <div class="file-chip">
          <span class="file-chip-name">${escapeHtml(file.name)}</span>
          <button type="button" class="file-chip-remove" onclick="removeSelectedImage(${index})">×</button>
        </div>
      `
    )
    .join("");
}

function setAnalyzerStatus(message) {
  const status = document.getElementById("analyzerStatus");
  if (status) {
    status.innerHTML = `<span class="inline-status">${escapeHtml(message)}</span>`;
  }
}

function loadSampleEmails() {
  const textarea = document.getElementById("emailText");
  textarea.value = [
    "From: admissions@qau.edu.pk\nSubject: Merit Scholarship Applications Open\nBody: Quaid-e-Azam University invites undergraduate students to apply for merit scholarships before 30 April 2026. Selected students will receive tuition support and mentorship.",
    "---",
    "From: info@nust.edu.pk\nSubject: Summer Research Internship 2026\nBody: Applications are open for the NUST summer research internship program. Students in CS, EE, and Data Science are encouraged to apply. Deadline: 25 April 2026."
  ].join("\n");
}

async function saveProfile() {
  const payload = {
    name: document.getElementById("name").value.trim(),
    university: document.getElementById("university").value.trim(),
    degree: document.getElementById("degree").value.trim(),
    semester: document.getElementById("semester").value.trim(),
    gpa: document.getElementById("gpa").value.trim(),
    preferences: document.getElementById("preferences").value.trim(),
    financial_need: document.getElementById("financial_need").value.trim(),
    past_experience: document.getElementById("past_experience").value.trim(),
  };

  const status = document.getElementById("profileStatus");
  status.innerHTML = `<span class="inline-status">Saving profile...</span>`;

  try {
    const res = await fetch(`${API_BASE}/submit-form`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Failed to save profile");
    }

    status.innerHTML = `<span class="inline-status">✅ Profile saved successfully</span>`;
  } catch (err) {
    status.innerHTML = `<span class="inline-status">❌ ${escapeHtml(err.message)}</span>`;
  }
}

async function analyzeEmails() {
  const btn = document.getElementById("analyzeBtn");
  const textarea = document.getElementById("emailText");

  const rawText = textarea.value.trim();
  const texts = rawText
    ? rawText
        .split(/\n-{3,}\n|\n---\n/g)
        .map((t) => t.trim())
        .filter(Boolean)
    : [];

  if (!texts.length && !state.selectedImages.length) {
    setAnalyzerStatus("Please paste text or attach at least one image.");
    return;
  }

  state.analyzing = true;
  btn.disabled = true;
  btn.innerText = "Analyzing...";
  setAnalyzerStatus("Sending data to Flask backend...");

  try {
    let res;

    if (state.selectedImages.length > 0) {
      const formData = new FormData();

      texts.forEach((text) => {
        formData.append("texts", text);
      });

      state.selectedImages.forEach((file) => {
        formData.append("images", file);
      });

      res = await fetch(`${API_BASE}/process-inputs`, {
        method: "POST",
        body: formData,
      });
    } else {
      res = await fetch(`${API_BASE}/process-inputs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts }),
      });
    }

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Analysis failed");
    }

    state.analyzed = true;
    state.lastAnalysis = data;

    renderDashboard(data);
    renderEduPredictions(data.edu_predictions || []);
    setAnalyzerStatus("✅ Analysis complete");
    switchPage("dashboard");
  } catch (err) {
    setAnalyzerStatus(`❌ ${err.message || "Backend not connected"}`);
  } finally {
    state.analyzing = false;
    btn.disabled = false;
    btn.innerText = "Analyze";
  }
}

function renderDashboard(data) {
  const container = document.getElementById("dashboardContent");

  const displayItems = data?.display_opportunities || [];
  const ranked = data?.ranked_output?.ranked_opportunities || [];
  const eduPredictions = data?.edu_predictions || [];
  const latestProfile = data?.latest_profile || null;

  document.getElementById("metricTotal").textContent = displayItems.length || ranked.length || 0;
  document.getElementById("metricAnalyzed").textContent = data?.input_count || 0;
  document.getElementById("metricEdu").textContent = eduPredictions.filter((x) => x.label === 1).length;
  document.getElementById("metricProfile").textContent = latestProfile ? "Yes" : "No";

  container.innerHTML = "";

  if (!displayItems.length) {
    if (ranked.length) {
      ranked.forEach((item, i) => {
        const combined = item?.combined_result?.combined || {};
        const score =
          item?.combined_result?.final_personalized_score ??
          item?.combined_result?.final_score ??
          0;
        const priorityClass = getPriorityClass(score);

        const card = document.createElement("div");
        card.className = `opp-card ${priorityClass} fade-in`;

        card.innerHTML = `
          <div class="flex-between mb-1">
            <div class="rank-num">#${i + 1}</div>
            <span class="badge badge-blue">Score ${Number(score).toFixed(2)}</span>
          </div>

          <h3 class="mb-1">${escapeHtml(combined.title || item.subject || "Untitled opportunity")}</h3>
          <div class="section-sub" style="margin-bottom:10px;">
            ${escapeHtml(combined.opportunity_type || "Other")} • Deadline: ${escapeHtml(combined.deadline_found || "Not mentioned")}
          </div>

          <p class="mb-1">${escapeHtml(combined.summary || "No summary available.")}</p>

          <div class="mt-1"><strong>Sender:</strong> ${escapeHtml(item.sender || "Unknown")}</div>
          <div class="mt-1"><strong>Location:</strong> ${escapeHtml(combined.location || "Not mentioned")}</div>
        `;

        container.appendChild(card);
      });
      return;
    }

    container.innerHTML = `
      <div class="card center-empty">
        No ranked opportunities were returned from the backend for this run.
      </div>
    `;
    return;
  }

  displayItems.forEach((opp, i) => {
    const score = Number(opp.score || 0);
    const priorityClass = getPriorityClass(score);

    const fitReason = Array.isArray(opp.fit_reason)
      ? opp.fit_reason.join(" | ")
      : (opp.fit_reason || "No reason available");

    const advisorReason =
      opp?.advisor_analysis?.why_it_matters ||
      opp?.advisor_analysis?.reason_for_decision ||
      "";

    const card = document.createElement("div");
    card.className = `opp-card ${priorityClass} fade-in`;

    card.innerHTML = `
      <div class="flex-between mb-1">
        <div class="rank-num">#${opp.rank || i + 1}</div>
        <span class="badge badge-blue">Score ${score.toFixed(2)}</span>
      </div>

      <h3 class="mb-1">${escapeHtml(opp.title || "Untitled opportunity")}</h3>

      <div class="section-sub" style="margin-bottom:10px;">
        ${escapeHtml(opp.type || "Other")} • Deadline: ${escapeHtml(opp.deadline || "Not mentioned")}
      </div>

      <p class="mb-1">${escapeHtml(opp.summary || "No summary available.")}</p>

      <div class="mt-1"><strong>Sender:</strong> ${escapeHtml(opp.sender || "Unknown")}</div>
      <div class="mt-1"><strong>Location:</strong> ${escapeHtml(opp.location || "Not mentioned")}</div>
      <div class="mt-1"><strong>Why it fits:</strong> ${escapeHtml(fitReason)}</div>
      ${
        advisorReason
          ? `<div class="mt-1"><strong>Personalized note:</strong> ${escapeHtml(advisorReason)}</div>`
          : ""
      }
    `;

    container.appendChild(card);
  });
}

function renderEduPredictions(predictions) {
  const container = document.getElementById("eduPredictions");
  container.innerHTML = "";

  if (!predictions.length) {
    container.innerHTML = `<div class="center-empty">No detection data yet.</div>`;
    return;
  }

  predictions.forEach((pred) => {
    const div = document.createElement("div");
    div.className = "email-item analyzed";

    const badgeClass = pred.label === 1 ? "badge-green" : "badge-red";

    div.innerHTML = `
      <div class="flex-between mb-1">
        <strong>${escapeHtml(pred.sender || "unknown")}</strong>
        <span class="badge ${badgeClass}">${escapeHtml(pred.prediction || "Unknown")}</span>
      </div>
      <div class="small-label">Confidence: ${escapeHtml(pred.confidence)}</div>
      <div>${escapeHtml(pred.body_preview || "")}</div>
    `;

    container.appendChild(div);
  });
}

function renderChat() {
  const chat = document.getElementById("chatBox");
  chat.innerHTML = "";

  state.chatMessages.forEach((message) => {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-bubble ${message.role}`;
    wrapper.innerText = message.text;
    chat.appendChild(wrapper);
  });

  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text) {
  state.chatMessages.push({ role, text });
  renderChat();
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const text = input.value.trim();

  if (!text) return;

  addMessage("user", text);
  input.value = "";

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "No response from chatbot");
    }

    addMessage("ai", data.reply || "No response");
  } catch (err) {
    addMessage("ai", `⚠️ ${err.message || "Backend not running"}`);
  }
}

function handleChatKey(event) {
  if (event.key === "Enter") {
    sendMessage();
  }
}

window.onload = () => {
  buildNav();
  buildApp();
  initializeAnalyzerUpload();
  switchPage("dashboard");
  renderDashboard({
    input_count: 0,
    edu_predictions: [],
    ranked_output: { ranked_opportunities: [] },
    display_opportunities: [],
    latest_profile: null,
  });
  renderChat();
};