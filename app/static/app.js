document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // DOM Elements
    const timeDisplay = document.getElementById("current-time");
    const totalDisplay = document.getElementById("stat-total");
    const logsTbody = document.getElementById("logs-tbody");
    const btnRefreshLogs = document.getElementById("btn-refresh-logs");
    
    // Settings Modal elements
    const settingsModal = document.getElementById("settings-modal");
    const navSettings = document.getElementById("nav-settings");
    const closeSettings = document.getElementById("close-settings");
    const settingsForm = document.getElementById("settings-form");
    const confSlider = document.getElementById("conf-slider");
    const confVal = document.getElementById("conf-val");
    const nmsSlider = document.getElementById("nms-slider");
    const nmsVal = document.getElementById("nms-val");
    const sourceInput = document.getElementById("source-input");

    // Preview Modal elements
    const previewModal = document.getElementById("preview-modal");
    const closePreview = document.getElementById("close-preview");
    const previewImg = document.getElementById("preview-img");
    const previewSeverity = document.getElementById("preview-severity");
    const previewTime = document.getElementById("preview-time");
    const previewConf = document.getElementById("preview-conf");

    // Chart Variables
    let severityChart = null;
    let timelineChart = null;

    // Time Clock
    function updateClock() {
        const now = new Date();
        timeDisplay.textContent = now.toLocaleTimeString();
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Chart Initializations
    function initCharts() {
        // Doughnut Chart for Severity
        const ctxSeverity = document.getElementById("severity-chart").getContext("2d");
        severityChart = new Chart(ctxSeverity, {
            type: "doughnut",
            data: {
                labels: ["Low", "Medium", "High"],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ["#10b981", "#f59e0b", "#f43f5e"],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: "#cbd5e1",
                            font: { family: "Inter", size: 12 }
                        }
                    }
                },
                cutout: "70%"
            }
        });

        // Line Chart for Timeline
        const ctxTimeline = document.getElementById("timeline-chart").getContext("2d");
        timelineChart = new Chart(ctxTimeline, {
            type: "line",
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                datasets: [{
                    label: "Detections",
                    data: Array(24).fill(0),
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99, 102, 241, 0.15)",
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: "#a855f7",
                    pointBorderColor: "#fff",
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", font: { family: "Inter", size: 10 } }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { 
                            color: "#94a3b8", 
                            font: { family: "Inter", size: 10 },
                            stepSize: 1,
                            precision: 0
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Fetch Stats & Logs
    async function fetchStats() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();

            // Total Counter
            totalDisplay.textContent = data.total;

            // Update Severity Chart
            if (severityChart) {
                severityChart.data.datasets[0].data = [
                    data.severity.Low || 0,
                    data.severity.Medium || 0,
                    data.severity.High || 0
                ];
                severityChart.update();
            }

            // Update Timeline Chart
            if (timelineChart) {
                const hours = Object.keys(data.hourly);
                const counts = Object.values(data.hourly);
                timelineChart.data.labels = hours.map(h => `${h}:00`);
                timelineChart.data.datasets[0].data = counts;
                timelineChart.update();
            }

            // Update Logs Table
            updateLogsTable(data.recent);

        } catch (err) {
            console.error("Error fetching stats:", err);
        }
    }

    function updateLogsTable(recentLogs) {
        if (!recentLogs || recentLogs.length === 0) {
            logsTbody.innerHTML = `
                <tr>
                    <td colspan="4" class="empty-state">No detections yet... Waiting for stream</td>
                </tr>
            `;
            return;
        }

        logsTbody.innerHTML = recentLogs.map(log => {
            const date = new Date(log.timestamp);
            const formattedTime = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            
            let badgeClass = "badge-success";
            if (log.severity === "High") badgeClass = "badge-danger";
            else if (log.severity === "Medium") badgeClass = "badge-warning";

            return `
                <tr>
                    <td>${formattedTime}</td>
                    <td>${Math.round(log.confidence * 100)}%</td>
                    <td><span class="badge ${badgeClass}">${log.severity}</span></td>
                    <td>
                        <button class="btn-link view-snapshot-btn" data-url="${log.image_path}" data-severity="${log.severity}" data-time="${date.toLocaleString()}" data-conf="${Math.round(log.confidence * 100)}">
                            View Image
                        </button>
                    </td>
                </tr>
            `;
        }).join("");

        // Attach event listeners to new snapshot buttons
        document.querySelectorAll(".view-snapshot-btn").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const target = e.target;
                previewImg.src = target.getAttribute("data-url");
                previewSeverity.textContent = target.getAttribute("data-severity");
                previewTime.textContent = `Time: ${target.getAttribute("data-time")}`;
                previewConf.textContent = `Confidence: ${target.getAttribute("data-conf")}%`;
                
                // Color severity badge in modal
                previewSeverity.className = "badge";
                const sev = target.getAttribute("data-severity");
                if (sev === "High") previewSeverity.classList.add("badge-danger");
                else if (sev === "Medium") previewSeverity.classList.add("badge-warning");
                else previewSeverity.classList.add("badge-success");

                previewModal.classList.add("active");
            });
        });
    }

    // Config Actions
    async function fetchConfig() {
        try {
            const res = await fetch("/api/config");
            const config = await res.json();

            confSlider.value = config.confidence_threshold;
            confVal.textContent = config.confidence_threshold;
            
            nmsSlider.value = config.nms_threshold;
            nmsVal.textContent = config.nms_threshold;

            sourceInput.value = config.video_source;
        } catch (err) {
            console.error("Error loading config:", err);
        }
    }

    // Modal Control
    navSettings.addEventListener("click", (e) => {
        e.preventDefault();
        fetchConfig();
        settingsModal.classList.add("active");
    });

    closeSettings.addEventListener("click", () => {
        settingsModal.classList.remove("active");
    });

    closePreview.addEventListener("click", () => {
        previewModal.classList.remove("active");
        previewImg.src = "";
    });

    // Close modals on clicking backdrop
    window.addEventListener("click", (e) => {
        if (e.target === settingsModal) settingsModal.classList.remove("active");
        if (e.target === previewModal) {
            previewModal.classList.remove("active");
            previewImg.src = "";
        }
    });

    // Handle Form submission
    settingsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const conf = confSlider.value;
        const nms = nmsSlider.value;
        const source = sourceInput.value;

        try {
            const res = await fetch(`/api/config?confidence_threshold=${conf}&nms_threshold=${nms}&video_source=${source}`, {
                method: "POST"
            });
            const result = await res.json();
            if (result.status === "success") {
                settingsModal.classList.remove("active");
                fetchStats();
            } else {
                alert("Failed to update configurations: " + result.message);
            }
        } catch (err) {
            console.error("Error updating config:", err);
            alert("Error communicating with server.");
        }
    });

    // Slider range update values
    confSlider.addEventListener("input", (e) => {
        confVal.textContent = e.target.value;
    });

    nmsSlider.addEventListener("input", (e) => {
        nmsVal.textContent = e.target.value;
    });

    // Table manual refresh
    btnRefreshLogs.addEventListener("click", () => {
        fetchStats();
    });

    // Menu active class toggling
    document.querySelectorAll(".nav-item").forEach(item => {
        if (item.id !== "nav-settings") {
            item.addEventListener("click", () => {
                document.querySelectorAll(".nav-item").forEach(i => i.classList.remove("active"));
                item.classList.add("active");
            });
        }
    });

    // Navigation links behaviors
    const navHistory = document.getElementById("nav-history");
    navHistory.addEventListener("click", (e) => {
        e.preventDefault();
        // Scroll to table card
        document.querySelector(".table-card").scrollIntoView({ behavior: "smooth" });
    });

    // Page initialization
    initCharts();
    fetchStats();
    fetchConfig();

    // Auto-update dashboard telemetry every 3 seconds
    setInterval(fetchStats, 3000);
});
