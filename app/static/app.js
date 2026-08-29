document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // ── Global App State ─────────────────────────────────────────────────
    let currentSelectedFile = null;
    let currentVideoData = null;
    let currentActiveFilter = "all";

    // ── DOM References ───────────────────────────────────────────────────
    // Navigation
    const navUpload = document.getElementById("nav-upload");
    const navHistory = document.getElementById("nav-history");
    const navStats = document.getElementById("nav-stats");
    const sectionUpload = document.getElementById("section-upload");
    const sectionHistory = document.getElementById("section-history");
    const sectionStats = document.getElementById("section-stats");
    const headerTitle = document.getElementById("header-title");
    const headerSubtitle = document.getElementById("header-subtitle");
    const liveClock = document.getElementById("live-clock");

    // Upload & Process Components
    const dropZone = document.getElementById("drop-zone");
    const videoInput = document.getElementById("video-input");
    const btnSelectFile = document.getElementById("btn-select-file");
    const fileInfoPanel = document.getElementById("file-info-panel");
    const selectedFileName = document.getElementById("selected-file-name");
    const selectedFileSize = document.getElementById("selected-file-size");
    const btnChangeFile = document.getElementById("btn-change-file");
    const btnStartProcess = document.getElementById("btn-start-process");
    const processingPanel = document.getElementById("processing-panel");
    const uploadContainer = document.getElementById("upload-container");

    // Results Components
    const resultsContainer = document.getElementById("results-container");
    const resultVideoPlayer = document.getElementById("result-video-player");
    const resVideoName = document.getElementById("res-video-name");
    const resVideoMeta = document.getElementById("res-video-meta");
    const kpiTotalPotholes = document.getElementById("kpi-total-potholes");
    const kpiDetectedFrames = document.getElementById("kpi-detected-frames");
    const kpiTotalFramesRatio = document.getElementById("kpi-total-frames-ratio");
    const kpiMaxConf = document.getElementById("kpi-max-conf");
    const kpiSevHigh = document.getElementById("kpi-sev-high");
    const kpiSevMedium = document.getElementById("kpi-sev-medium");
    const kpiSevLow = document.getElementById("kpi-sev-low");
    const barSevHigh = document.getElementById("bar-sev-high");
    const barSevMedium = document.getElementById("bar-sev-medium");
    const barSevLow = document.getElementById("bar-sev-low");
    const framesGrid = document.getElementById("frames-grid");
    const emptyGallery = document.getElementById("empty-gallery");
    const galleryTotalBadge = document.getElementById("gallery-total-badge");
    const filterButtons = document.querySelectorAll(".filter-btn");

    // History Components
    const historyTbody = document.getElementById("history-tbody");
    const btnRefreshHistory = document.getElementById("btn-refresh-history");

    // Stats Components
    const statTotalVideos = document.getElementById("stat-total-videos");
    const statTotalPotholes = document.getElementById("stat-total-potholes");
    const statTotalFrames = document.getElementById("stat-total-frames");
    const overallSeverityRow = document.getElementById("overall-severity-row");

    // Modals
    const modalPreview = document.getElementById("modal-preview");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const modalImg = document.getElementById("modal-img");
    const modalFrameTitle = document.getElementById("modal-frame-title");
    const modalFrameBadge = document.getElementById("modal-frame-badge");
    const modalTimestamp = document.getElementById("modal-timestamp");
    const modalPotholeCount = document.getElementById("modal-pothole-count");
    const modalMaxConf = document.getElementById("modal-max-conf");
    const modalConfList = document.getElementById("modal-conf-list");

    const modalReset = document.getElementById("modal-reset");
    const btnResetDb = document.getElementById("btn-reset-db");
    const btnCloseResetModal = document.getElementById("btn-close-reset-modal");
    const btnCancelReset = document.getElementById("btn-cancel-reset");
    const btnConfirmReset = document.getElementById("btn-confirm-reset");

    // ── Clock ────────────────────────────────────────────────────────────
    function updateLiveClock() {
        const now = new Date();
        liveClock.textContent = now.toLocaleTimeString("vi-VN");
    }
    setInterval(updateLiveClock, 1000);
    updateLiveClock();

    // ── Navigation Tabs ──────────────────────────────────────────────────
    function switchTab(tabId) {
        [navUpload, navHistory, navStats].forEach(n => n.classList.remove("active"));
        [sectionUpload, sectionHistory, sectionStats].forEach(s => s.style.display = "none");

        if (tabId === "upload") {
            navUpload.classList.add("active");
            sectionUpload.style.display = "block";
            headerTitle.textContent = "Nhận diện Ổ gà từ Video";
            headerSubtitle.textContent = "Tải video lên để mô hình AI tự động quét từng khung hình và trích xuất các vị trí ổ gà";
        } else if (tabId === "history") {
            navHistory.classList.add("active");
            sectionHistory.style.display = "block";
            headerTitle.textContent = "Lịch sử Phân tích Video";
            headerSubtitle.textContent = "Xem lại toàn bộ video đã xử lý và các khung hình ổ gà đã lưu trữ";
            loadHistoryTable();
        } else if (tabId === "stats") {
            navStats.classList.add("active");
            sectionStats.style.display = "block";
            headerTitle.textContent = "Thống kê Hệ thống AI";
            headerSubtitle.textContent = "Tổng hợp dữ liệu nhận diện và phân bố mức độ nghiêm trọng";
            loadOverallStats();
        }
        lucide.createIcons();
    }

    navUpload.addEventListener("click", () => switchTab("upload"));
    navHistory.addEventListener("click", () => switchTab("history"));
    navStats.addEventListener("click", () => switchTab("stats"));

    // ── File Selection & Drag Drop ───────────────────────────────────────
    function handleSelectedFile(file) {
        if (!file) return;
        currentSelectedFile = file;
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        dropZone.style.display = "none";
        fileInfoPanel.style.display = "flex";
        lucide.createIcons();
    }

    dropZone.addEventListener("click", () => videoInput.click());
    btnSelectFile.addEventListener("click", (e) => {
        e.stopPropagation();
        videoInput.click();
    });

    videoInput.addEventListener("change", () => {
        if (videoInput.files && videoInput.files[0]) {
            handleSelectedFile(videoInput.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary)";
        dropZone.style.background = "rgba(99, 102, 241, 0.08)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "";
        dropZone.style.background = "";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "";
        dropZone.style.background = "";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    btnChangeFile.addEventListener("click", () => {
        currentSelectedFile = null;
        videoInput.value = "";
        fileInfoPanel.style.display = "none";
        dropZone.style.display = "flex";
        lucide.createIcons();
    });

    // ── Upload & Process Video ───────────────────────────────────────────
    btnStartProcess.addEventListener("click", async () => {
        if (!currentSelectedFile) return;

        fileInfoPanel.style.display = "none";
        processingPanel.style.display = "flex";
        resultsContainer.style.display = "none";

        const formData = new FormData();
        formData.append("file", currentSelectedFile);

        try {
            const response = await fetch("/api/upload-video", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Upload và xử lý thất bại");
            }

            const result = await response.json();
            currentVideoData = result.data;

            // Render Results
            renderAnalysisResults(currentVideoData);

            // Reset upload form state
            processingPanel.style.display = "none";
            dropZone.style.display = "flex";
            videoInput.value = "";
            currentSelectedFile = null;

        } catch (error) {
            alert(`Lỗi: ${error.message}`);
            processingPanel.style.display = "none";
            fileInfoPanel.style.display = "flex";
        }
    });

    // ── Render Analysis Results ──────────────────────────────────────────
    function renderAnalysisResults(videoData) {
        if (!videoData) return;

        // 1. Video Player
        resultVideoPlayer.src = videoData.video_url;
        resultVideoPlayer.load();
        resVideoName.textContent = videoData.filename;
        resVideoMeta.textContent = `${videoData.duration_sec}s · ${videoData.fps} FPS · ${videoData.total_frames} Frames`;

        // 2. KPI Summary
        kpiTotalPotholes.textContent = videoData.potholes_count;
        kpiDetectedFrames.textContent = videoData.detected_frames_count;
        kpiTotalFramesRatio.textContent = `/ ${videoData.total_frames} frames`;

        // Calculate severity and max confidence
        const frames = videoData.frames || [];
        let maxOverallConf = 0;
        let highCount = 0;
        let mediumCount = 0;
        let lowCount = 0;

        frames.forEach(f => {
            if (f.max_confidence > maxOverallConf) maxOverallConf = f.max_confidence;
            if (f.severity === "High") highCount++;
            else if (f.severity === "Medium") mediumCount++;
            else if (f.severity === "Low") lowCount++;
        });

        kpiMaxConf.textContent = `${Math.round(maxOverallConf * 100)}%`;
        kpiSevHigh.textContent = highCount;
        kpiSevMedium.textContent = mediumCount;
        kpiSevLow.textContent = lowCount;

        const totalDetected = frames.length || 1;
        barSevHigh.style.width = `${(highCount / totalDetected) * 100}%`;
        barSevMedium.style.width = `${(mediumCount / totalDetected) * 100}%`;
        barSevLow.style.width = `${(lowCount / totalDetected) * 100}%`;

        // 3. Render Frames Gallery
        galleryTotalBadge.textContent = `${frames.length} ảnh`;
        renderFramesGrid(frames);

        // Show Results Section
        resultsContainer.style.display = "block";
        resultsContainer.scrollIntoView({ behavior: "smooth" });
        lucide.createIcons();
    }

    // ── Render Frames Grid ───────────────────────────────────────────────
    function renderFramesGrid(frames) {
        framesGrid.innerHTML = "";

        const filtered = currentActiveFilter === "all" 
            ? frames 
            : frames.filter(f => f.severity.toLowerCase() === currentActiveFilter.toLowerCase());

        if (filtered.length === 0) {
            emptyGallery.style.display = "flex";
            return;
        } else {
            emptyGallery.style.display = "none";
        }

        filtered.forEach(frame => {
            const card = document.createElement("div");
            card.className = "frame-card";

            const badgeClass = frame.severity === "High" ? "badge-danger" 
                             : frame.severity === "Medium" ? "badge-warning" 
                             : "badge-success";

            // Render confidence chips for each pothole detected in this frame
            const confs = frame.confidences || [];
            const confChipsHtml = confs.map((c, idx) => {
                const pct = Math.round(c * 100);
                const chipClass = pct >= 80 ? "chip-high" : pct >= 50 ? "chip-medium" : "chip-low";
                return `<span class="conf-chip ${chipClass}">#${idx + 1}: ${pct}%</span>`;
            }).join("");

            card.innerHTML = `
                <div class="frame-thumbnail">
                    <img src="${frame.image_path}" alt="Frame #${frame.frame_number}" loading="lazy">
                    <div class="frame-hover-overlay">
                        <i data-lucide="maximize-2"></i>
                        <span>Xem chi tiết</span>
                    </div>
                </div>
                <div class="frame-card-body">
                    <div class="frame-row-top">
                        <span class="frame-num">Frame #${frame.frame_number}</span>
                        <span class="badge ${badgeClass}">${frame.severity}</span>
                    </div>
                    <div class="frame-row-top">
                        <span class="frame-time">⏱️ ${frame.timestamp_sec}s</span>
                        <span class="text-emerald" style="font-weight: 600; font-size: 0.85rem;">Max: ${Math.round(frame.max_confidence * 100)}%</span>
                    </div>
                    <div class="conf-badges-wrap">
                        ${confChipsHtml}
                    </div>
                </div>
            `;

            card.addEventListener("click", () => openFrameModal(frame));
            framesGrid.appendChild(card);
        });

        lucide.createIcons();
    }

    // ── Filter Buttons ───────────────────────────────────────────────────
    filterButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            filterButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentActiveFilter = btn.getAttribute("data-filter");
            if (currentVideoData && currentVideoData.frames) {
                renderFramesGrid(currentVideoData.frames);
            }
        });
    });

    // ── Frame Detail Modal ───────────────────────────────────────────────
    function openFrameModal(frame) {
        modalImg.src = frame.image_path;
        modalFrameTitle.textContent = `Khung hình #${frame.frame_number}`;
        modalTimestamp.textContent = `${frame.timestamp_sec} giây`;
        modalPotholeCount.textContent = `${frame.pothole_count} ổ gà`;
        modalMaxConf.textContent = `${Math.round(frame.max_confidence * 100)}%`;

        modalFrameBadge.className = `badge ${frame.severity === "High" ? "badge-danger" : frame.severity === "Medium" ? "badge-warning" : "badge-success"}`;
        modalFrameBadge.textContent = frame.severity;

        // Render conf list
        const confs = frame.confidences || [];
        modalConfList.innerHTML = confs.map((c, idx) => {
            const pct = Math.round(c * 100);
            return `
                <div class="modal-conf-item">
                    <span>Ổ gà #${idx + 1}</span>
                    <strong class="text-emerald">${pct}%</strong>
                </div>
            `;
        }).join("");

        modalPreview.classList.add("active");
        lucide.createIcons();
    }

    btnCloseModal.addEventListener("click", () => modalPreview.classList.remove("active"));
    modalPreview.addEventListener("click", (e) => {
        if (e.target === modalPreview) modalPreview.classList.remove("active");
    });

    // ── History Table Loader ─────────────────────────────────────────────
    async function loadHistoryTable() {
        historyTbody.innerHTML = `<tr><td colspan="8" class="text-center py-4">Đang tải danh sách video...</td></tr>`;
        try {
            const res = await fetch("/api/videos");
            const videos = await res.json();

            if (!videos || videos.length === 0) {
                historyTbody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-muted">Chưa có video nào được xử lý. Hãy tải lên video mới!</td></tr>`;
                return;
            }

            historyTbody.innerHTML = videos.map(v => {
                const dt = v.uploaded_at ? new Date(v.uploaded_at).toLocaleString("vi-VN") : "N/A";
                return `
                    <tr>
                        <td><strong>#${v.id}</strong></td>
                        <td>${v.filename}</td>
                        <td>${v.duration_sec}s</td>
                        <td>${v.total_frames}</td>
                        <td><span class="badge badge-danger">${v.potholes_count} ổ gà</span></td>
                        <td>${v.detected_frames_count} frames</td>
                        <td>${dt}</td>
                        <td>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn btn-secondary btn-sm" onclick="viewHistoryVideo(${v.id})">
                                    <i data-lucide="eye"></i> Xem lại
                                </button>
                                <button class="btn btn-outline-danger btn-sm" onclick="deleteHistoryVideo(${v.id})">
                                    <i data-lucide="trash"></i> Xóa
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }).join("");

            lucide.createIcons();
        } catch (err) {
            historyTbody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-rose">Lỗi khi tải lịch sử: ${err.message}</td></tr>`;
        }
    }

    btnRefreshHistory.addEventListener("click", loadHistoryTable);

    // Global helper to view historical video
    window.viewHistoryVideo = async function(videoId) {
        try {
            const res = await fetch(`/api/videos/${videoId}`);
            if (!res.ok) throw new Error("Không thể tải thông tin video");
            const videoData = await res.json();
            currentVideoData = videoData;
            switchTab("upload");
            renderAnalysisResults(videoData);
        } catch (err) {
            alert(`Lỗi: ${err.message}`);
        }
    };

    // Global helper to delete historical video
    window.deleteHistoryVideo = async function(videoId) {
        if (!confirm(`Bạn có chắc chắn muốn xóa video #${videoId} cùng toàn bộ ảnh liên quan?`)) return;
        try {
            const res = await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
            if (!res.ok) throw new Error("Xóa video thất bại");
            loadHistoryTable();
        } catch (err) {
            alert(`Lỗi: ${err.message}`);
        }
    };

    // ── Overall Stats Loader ─────────────────────────────────────────────
    async function loadOverallStats() {
        try {
            const res = await fetch("/api/stats");
            const stats = await res.json();

            statTotalVideos.textContent = stats.total_videos || 0;
            statTotalPotholes.textContent = stats.total_potholes || 0;
            statTotalFrames.textContent = stats.total_detected_frames || 0;

            const sev = stats.severity || { High: 0, Medium: 0, Low: 0 };
            overallSeverityRow.innerHTML = `
                <div class="card p-3">
                    <span class="text-muted">🔴 Mức độ Cao (Nghiêm trọng)</span>
                    <h2 class="text-rose mt-2">${sev.High} khung hình</h2>
                </div>
                <div class="card p-3">
                    <span class="text-muted">🟠 Mức độ Vừa (Cần sửa)</span>
                    <h2 class="text-amber mt-2">${sev.Medium} khung hình</h2>
                </div>
                <div class="card p-3">
                    <span class="text-muted">🟢 Mức độ Nhẹ (Rạn nứt)</span>
                    <h2 class="text-emerald mt-2">${sev.Low} khung hình</h2>
                </div>
            `;
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    }

    // ── Database Reset Confirmation ──────────────────────────────────────
    btnResetDb.addEventListener("click", () => modalReset.classList.add("active"));
    btnCloseResetModal.addEventListener("click", () => modalReset.classList.remove("active"));
    btnCancelReset.addEventListener("click", () => modalReset.classList.remove("active"));

    btnConfirmReset.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/reset-db", { method: "POST" });
            const data = await res.json();
            modalReset.classList.remove("active");
            alert(data.message || "Database đã được reset thành công!");
            
            // Clear current view
            currentVideoData = null;
            resultsContainer.style.display = "none";
            fileInfoPanel.style.display = "none";
            dropZone.style.display = "flex";
            
            switchTab("upload");
        } catch (err) {
            alert(`Lỗi reset: ${err.message}`);
        }
    });

});
