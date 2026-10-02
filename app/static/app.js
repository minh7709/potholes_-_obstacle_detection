document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // ── Safe DOM helpers ─────────────────────────────────────────────────
    // getElementById returns null when an element is absent, and assigning to
    // .textContent on null throws, which blanks the whole page. That is not a
    // cosmetic problem: it silently killed every handler on the page whenever
    // the HTML and this script drifted out of sync (a renamed KPI id is enough).
    // setText degrades to a no-op so a partial mismatch cannot take the app down.
    const setText = (element, value) => {
        if (element) element.textContent = value;
    };
    const setHtml = (element, value) => {
        if (element) element.innerHTML = value;
    };
    const setStyle = (element, prop, value) => {
        if (element && element.style) element.style[prop] = value;
    };

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
    const kpiTotalWarnings = document.getElementById("kpi-total-warnings");
    const kpiMinTtc = document.getElementById("kpi-min-ttc");
    const kpiMinDistance = document.getElementById("kpi-min-distance");
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
    const statTotalWarnings = document.getElementById("stat-total-potholes");
    const statAvgTtc = document.getElementById("stat-total-frames");
    const overallSeverityRow = document.getElementById("overall-severity-row");

    // Modals
    const modalPreview = document.getElementById("modal-preview");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const modalImg = document.getElementById("modal-img");
    const modalFrameTitle = document.getElementById("modal-frame-title");
    const modalFrameBadge = document.getElementById("modal-frame-badge");
    const modalTimestamp = document.getElementById("modal-timestamp");
    const modalObject = document.getElementById("modal-pothole-count");
    const modalDistance = document.getElementById("modal-max-conf");
    const modalClosingSpeed = document.getElementById("modal-closing-speed");
    const modalTtc = document.getElementById("modal-ttc");
    const modalEgoSpeed = document.getElementById("modal-ego-speed");
    const modalReason = document.getElementById("modal-reason");
    const modalConfList = document.getElementById("modal-conf-list");

    const modalReset = document.getElementById("modal-reset");
    const btnResetDb = document.getElementById("btn-reset-db");
    const btnCloseResetModal = document.getElementById("btn-close-reset-modal");
    const btnCancelReset = document.getElementById("btn-cancel-reset");
    const btnConfirmReset = document.getElementById("btn-confirm-reset");

    // ── Clock ────────────────────────────────────────────────────────────
    function updateLiveClock() {
        const now = new Date();
        setText(liveClock, now.toLocaleTimeString("vi-VN"));
    }
    setInterval(updateLiveClock, 1000);
    updateLiveClock();

    // ── Navigation Tabs ──────────────────────────────────────────────────
    function switchTab(tabId) {
        [navUpload, navHistory, navStats].forEach(n => n.classList.remove("active"));
        [sectionUpload, sectionHistory, sectionStats].forEach(s => s.style.display = "none");

        if (tabId === "upload") {
            navUpload.classList.add("active");
            setStyle(sectionUpload, "display", "block");
            setText(headerTitle, "Cảnh báo Va chạm từ Video");
            setText(headerSubtitle, "Tải video lên để AI nhận diện xe phía trước, ước lượng khoảng cách, tốc độ đóng và thời gian va chạm (TTC)");
        } else if (tabId === "history") {
            navHistory.classList.add("active");
            setStyle(sectionHistory, "display", "block");
            setText(headerTitle, "Lịch sử Phân tích Video");
            setText(headerSubtitle, "Xem lại toàn bộ video đã xử lý và các khung hình cảnh báo đã lưu trữ");
            loadHistoryTable();
        } else if (tabId === "stats") {
            navStats.classList.add("active");
            setStyle(sectionStats, "display", "block");
            setText(headerTitle, "Thống kê Hệ thống AI");
            setText(headerSubtitle, "Tổng hợp dữ liệu cảnh báo va chạm và phân bố mức nguy hiểm");
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
        setText(selectedFileName, file.name);
        setText(selectedFileSize, `${(file.size / (1024 * 1024)).toFixed(2)} MB`);

        setStyle(dropZone, "display", "none");
        setStyle(fileInfoPanel, "display", "flex");
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
        setStyle(dropZone, "borderColor", "var(--primary)");
        setStyle(dropZone, "background", "rgba(99, 102, 241, 0.08)");
    });

    dropZone.addEventListener("dragleave", () => {
        setStyle(dropZone, "borderColor", "");
        setStyle(dropZone, "background", "");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        setStyle(dropZone, "borderColor", "");
        setStyle(dropZone, "background", "");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    btnChangeFile.addEventListener("click", () => {
        currentSelectedFile = null;
        videoInput.value = "";
        setStyle(fileInfoPanel, "display", "none");
        setStyle(dropZone, "display", "flex");
        lucide.createIcons();
    });

    // ── Upload & Process Video ───────────────────────────────────────────
    btnStartProcess.addEventListener("click", async () => {
        if (!currentSelectedFile) return;

        setStyle(fileInfoPanel, "display", "none");
        setStyle(processingPanel, "display", "flex");
        setStyle(resultsContainer, "display", "none");

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
            setStyle(processingPanel, "display", "none");
            setStyle(dropZone, "display", "flex");
            videoInput.value = "";
            currentSelectedFile = null;

        } catch (error) {
            alert(`Lỗi: ${error.message}`);
            setStyle(processingPanel, "display", "none");
            setStyle(fileInfoPanel, "display", "flex");
        }
    });

    // ── Render Analysis Results ──────────────────────────────────────────
    function renderAnalysisResults(videoData) {
        if (!videoData) return;

        // 1. Video Player
        resultVideoPlayer.src = videoData.video_url;
        resultVideoPlayer.load();
        setText(resVideoName, videoData.filename);
        setText(resVideoMeta, `${videoData.duration_sec}s · ${videoData.fps} FPS · ${videoData.total_frames} Frames`);

        // 2. KPI Summary — distance and TTC are ESTIMATES from monocular depth.
        setText(kpiTotalWarnings, videoData.warning_frames_count || 0);
        setText(kpiMinTtc, videoData.min_ttc_sec != null ? `${videoData.min_ttc_sec}s` : "—");
        setText(kpiMinDistance, videoData.min_distance_m != null ? `${videoData.min_distance_m} m` : "—");

        // Split stored frames by warning level.
        const frames = videoData.frames || [];
        let dangerCount = 0;
        let warningCount = 0;
        frames.forEach(f => {
            if (f.severity === "DANGER") dangerCount++;
            else warningCount++;
        });

        setText(kpiSevHigh, dangerCount);
        setText(kpiSevMedium, warningCount);
        setText(kpiSevLow, 0);  // Normal frames are not persisted to keep the DB small

        const totalStored = frames.length || 1;
        setStyle(barSevHigh, "width", `${(dangerCount / totalStored) * 100}%`);
        setStyle(barSevMedium, "width", `${(warningCount / totalStored) * 100}%`);
        setStyle(barSevLow, "width", "0%");

        // 3. Render Warning Frames Gallery
        setText(galleryTotalBadge, `${frames.length} ảnh`);
        renderFramesGrid(frames);

        // Show Results Section
        setStyle(resultsContainer, "display", "block");
        resultsContainer.scrollIntoView({ behavior: "smooth" });
        lucide.createIcons();
    }

    // ── Render Frames Grid ───────────────────────────────────────────────
    function renderFramesGrid(frames) {
        setHtml(framesGrid, "");

        const filtered = currentActiveFilter === "all"
            ? frames
            : frames.filter(f => f.severity === currentActiveFilter);

        if (filtered.length === 0) {
            setStyle(emptyGallery, "display", "flex");
            return;
        } else {
            setStyle(emptyGallery, "display", "none");
        }

        filtered.forEach(frame => {
            const card = document.createElement("div");
            card.className = "frame-card";

            const badgeClass = frame.severity === "DANGER" ? "badge-danger" : "badge-warning";

            const dist = frame.distance_m != null ? `${frame.distance_m} m` : "—";
            const ttc = frame.ttc_sec != null ? `${frame.ttc_sec}s` : "—";
            const speed = frame.closing_speed_mps != null ? `${frame.closing_speed_mps} m/s` : "—";

            setHtml(card, `
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
                        <span class="text-rose" style="font-weight: 600; font-size: 0.85rem;">TTC: ${ttc}</span>
                    </div>
                    <div class="conf-badges-wrap">
                        <span class="conf-chip chip-high">${frame.class_name} #${frame.track_id ?? "—"}</span>
                        <span class="conf-chip chip-medium">${dist}</span>
                        <span class="conf-chip chip-low">${speed}</span>
                    </div>
                </div>
            `);

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
        setText(modalFrameTitle, `Khung hình #${frame.frame_number}`);
        setText(modalTimestamp, `${frame.timestamp_sec} giây`);
        setText(modalObject, `${frame.class_name} #${frame.track_id ?? "—"}`);
        setText(modalDistance, frame.distance_m != null ? `${frame.distance_m} m` : "—");
        setText(modalClosingSpeed, frame.closing_speed_mps != null ? `${frame.closing_speed_mps} m/s` : "—");
        setText(modalTtc, frame.ttc_sec != null ? `${frame.ttc_sec} s` : "—");
        setText(modalEgoSpeed, `${frame.ego_speed_mps} m/s`);

        // Range and TTC warn independently, so say which one fired. A DANGER with
        // TTC = infinity otherwise looks like a bug rather than the close-range rule.
        const primaryDet = (frame.detections || []).find(o => o.track_id === frame.track_id)
                        || (frame.detections || [])[0];
        let reason = "—";
        if (primaryDet && primaryDet.proximity_trigger) {
            reason = "QUÁ GẦN (TTC không áp dụng)";
        } else if (frame.severity === "DANGER") {
            reason = "TTC ≤ ngưỡng nguy hiểm";
        } else if (frame.severity === "WARNING") {
            reason = "TTC ≤ ngưỡng cảnh báo";
        }
        setText(modalReason, reason);
        if (modalReason) {
            modalReason.className = (primaryDet && primaryDet.proximity_trigger)
                ? "text-rose" : "text-amber";
        }

        modalFrameBadge.className = `badge ${frame.severity === "DANGER" ? "badge-danger" : "badge-warning"}`;
        setText(modalFrameBadge, frame.severity);

        // All tracked objects in this frame with their telemetry.
        const objects = frame.detections || [];
        setHtml(modalConfList, objects.length ? objects.map(o => `
            <div class="modal-conf-item">
                <span>${o.class} #${o.track_id ?? "—"}</span>
                <strong class="${o.level === "DANGER" ? "text-rose" : o.level === "WARNING" ? "text-amber" : "text-emerald"}">
                    ${o.distance_m != null ? o.distance_m + "m" : "—"} · TTC ${o.ttc_s != null ? o.ttc_s + "s" : "∞"}
                </strong>
            </div>
        `).join("") : '<p class="text-muted">Không có đối tượng nào được theo dõi.</p>');

        modalPreview.classList.add("active");
        lucide.createIcons();
    }

    btnCloseModal.addEventListener("click", () => modalPreview.classList.remove("active"));
    modalPreview.addEventListener("click", (e) => {
        if (e.target === modalPreview) modalPreview.classList.remove("active");
    });

    // ── History Table Loader ─────────────────────────────────────────────
    async function loadHistoryTable() {
        setHtml(historyTbody, `<tr><td colspan="9" class="text-center py-4">Đang tải danh sách video...</td></tr>`);
        try {
            const res = await fetch("/api/videos");
            const videos = await res.json();

            if (!videos || videos.length === 0) {
                setHtml(historyTbody, `<tr><td colspan="9" class="text-center py-4 text-muted">Chưa có video nào được xử lý. Hãy tải lên video mới!</td></tr>`);
                return;
            }

            setHtml(historyTbody, videos.map(v => {
                const dt = v.uploaded_at ? new Date(v.uploaded_at).toLocaleString("vi-VN") : "N/A";
                const ttc = v.min_ttc_sec != null ? `${v.min_ttc_sec}s` : "—";
                const dangerBadge = v.danger_frames_count > 0
                    ? `<span class="badge badge-danger">${v.danger_frames_count} NGUY HIỂM</span>`
                    : `<span class="badge badge-success">0</span>`;
                return `
                    <tr>
                        <td><strong>#${v.id}</strong></td>
                        <td>${v.filename}</td>
                        <td>${v.duration_sec}s</td>
                        <td>${v.total_frames}</td>
                        <td><span class="badge badge-warning">${v.warning_frames_count} khung</span></td>
                        <td>${dangerBadge}</td>
                        <td><strong class="${v.min_ttc_sec != null && v.min_ttc_sec <= 2 ? "text-rose" : ""}">${ttc}</strong></td>
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
            }).join(""));

            lucide.createIcons();
        } catch (err) {
            setHtml(historyTbody, `<tr><td colspan="9" class="text-center py-4 text-rose">Lỗi khi tải lịch sử: ${err.message}</td></tr>`);
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

            setText(statTotalVideos, stats.total_videos || 0);
            setText(statTotalWarnings, stats.total_warning_frames || 0);
            setText(statAvgTtc, stats.avg_danger_ttc_sec != null ? `${stats.avg_danger_ttc_sec}s` : "—");

            const sev = stats.severity || { DANGER: 0, WARNING: 0, NORMAL: 0 };
            setHtml(overallSeverityRow, `
                <div class="card p-3">
                    <span class="text-muted">🔴 NGUY HIỂM (TTC ≤ 2s)</span>
                    <h2 class="text-rose mt-2">${sev.DANGER} khung hình</h2>
                </div>
                <div class="card p-3">
                    <span class="text-muted">🟠 CẢNH BÁO (2s &lt; TTC ≤ 4s)</span>
                    <h2 class="text-amber mt-2">${sev.WARNING} khung hình</h2>
                </div>
                <div class="card p-3">
                    <span class="text-muted">🟢 BÌNH THƯỜNG (TTC &gt; 4s)</span>
                    <h2 class="text-emerald mt-2">${stats.total_videos ? "—" : "0"} khung hình</h2>
                    <span class="text-muted" style="font-size: 0.75rem;">Không lưu frame bình thường</span>
                </div>
            `);
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
            setStyle(resultsContainer, "display", "none");
            setStyle(fileInfoPanel, "display", "none");
            setStyle(dropZone, "display", "flex");
            
            switchTab("upload");
        } catch (err) {
            alert(`Lỗi reset: ${err.message}`);
        }
    });

});
