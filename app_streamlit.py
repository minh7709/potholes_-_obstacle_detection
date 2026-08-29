import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
from urllib.parse import quote

# Backend API server URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RoadVision - Pothole Detection & Telemetry System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Theme Custom styling
st.markdown("""
<style>
    /* Dark premium canvas background */
    .stApp {
        background-color: #080a15;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.06) 0px, transparent 50%);
    }
    
    /* Title text styling */
    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }
    
    /* System Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }
    
    /* Glassmorphic Metrics Card */
    .metric-card {
        background: rgba(16, 20, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(16px);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.25rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if backend server is online
def is_backend_online():
    try:
        response = requests.get(f"{API_URL}/api/config", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Setup Sidebar System Control Panel
st.sidebar.markdown('<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2rem;">'
                    '<div style="width: 32px; height: 32px; background: linear-gradient(135deg, #6366f1, #a855f7); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(99,102,241,0.3);">'
                    '<span style="color: white; font-weight: bold; font-size: 1.1rem;">R</span></div>'
                    '<span style="font-size: 1.5rem; font-weight: 800; font-family: \'Outfit\', sans-serif; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RoadVision</span>'
                    '</div>', unsafe_allow_html=True)

backend_online = is_backend_online()

if backend_online:
    st.sidebar.markdown('<div class="status-badge"><div class="status-dot"></div>Analyzing Live Stream</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge" style="background: rgba(244, 63, 94, 0.12); border-color: rgba(244, 63, 94, 0.25); color: #fda4af;">'
                        '<div class="status-dot" style="background-color: #f43f5e; box-shadow: 0 0 8px #f43f5e;"></div>Offline</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Detector Settings")

# Fetch configuration values from API
config_data = {"confidence_threshold": 0.25, "nms_threshold": 0.45, "video_source": "0"}
if backend_online:
    try:
        config_response = requests.get(f"{API_URL}/api/config")
        if config_response.status_code == 200:
            config_data = config_response.json()
    except Exception as e:
        st.sidebar.error(f"Error fetching config: {e}")

# Form to hold slider states
with st.sidebar.form("config_form"):
    conf_val = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=float(config_data["confidence_threshold"]), step=0.05)
    nms_val = st.slider("NMS Threshold (Overlap)", min_value=0.1, max_value=0.9, value=float(config_data["nms_threshold"]), step=0.05)
    source_val = st.text_input(
        "Stream Video Source", 
        value=str(config_data["video_source"]),
        help="Nhập '0' cho webcam máy tính, hoặc link MJPEG từ điện thoại (ví dụ: http://192.168.1.5:8080/stream.mjpg)."
    )
    submit_config = st.form_submit_button("Apply Configuration")

    if submit_config and backend_online:
        try:
            post_url = f"{API_URL}/api/config?confidence_threshold={conf_val}&nms_threshold={nms_val}&video_source={quote(source_val, safe='')}"
            res = requests.post(post_url)
            if res.status_code == 200:
                st.sidebar.success("Configuration updated!")
                # Give a short sleep for settings to apply
                time.sleep(0.5)
                st.rerun()
            else:
                st.sidebar.error(f"Failed to update config: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Communication error: {e}")

st.sidebar.markdown("---")
live_reload = st.sidebar.checkbox("🔄 Live Autorefresh", value=True)
if live_reload:
    st.sidebar.caption("Auto-refreshing dashboard metrics every 3 seconds.")

# Main Page Layout
st.markdown("<h1 style='margin-bottom: 5px;'>Pothole Detection & Telemetry</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;'>Real-time AI roadway inspection camera and statistics logs</p>", unsafe_allow_html=True)

if not backend_online:
    st.error("⚠️ Backend API Server is Offline. Please start the backend server by running `python main.py` first.")
    st.info("💡 Make sure you are running the command in the virtual environment `.venv\\Scripts\\activate`.")
    # Auto-retry every 3 seconds — do NOT call st.stop() or the page will never self-recover
    with st.spinner("Waiting for backend to come online..."):
        time.sleep(3)
    st.rerun()

# Load Statistics from Backend API
stats = {"total": 0, "severity": {"Low": 0, "Medium": 0, "High": 0}, "hourly": {}, "recent": []}
try:
    stats_response = requests.get(f"{API_URL}/api/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
except Exception as e:
    st.error(f"Error connecting to backend API: {e}")

# Grid Layout: Stream Feed & Metrics
col_stream, col_telemetry = st.columns([2, 1])

with col_stream:
    st.markdown("### 📹 Live Analysis Stream Feed")
    # Stream embedded MJPEG from backend
    st.markdown(
        f'<div style="display: flex; width: 100%; justify-content: center; background: #000; border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.07); aspect-ratio: 16/9;">'
        f'  <img src="{API_URL}/api/video_feed" style="width: 100%; object-fit: contain;">'
        f'</div>',
        unsafe_allow_html=True
    )

with col_telemetry:
    st.markdown("### 📊 Live Telemetry")
    
    # Render custom metric card for Total potholes
    st.markdown(
        f'<div class="metric-card">'
        f'  <div class="metric-label">Total Potholes Detected</div>'
        f'  <div class="metric-value">{stats["total"]}</div>'
        f'  <div class="metric-sub">Across all inspection streams</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Render severity metrics breakdown
    st.markdown("#### ⚠️ Severity Distribution")
    sev_data = stats["severity"]
    
    # Create simple data table
    sev_df = pd.DataFrame({
        "Severity": ["Low", "Medium", "High"],
        "Count": [sev_data.get("Low", 0), sev_data.get("Medium", 0), sev_data.get("High", 0)]
    })
    
    # Color-coded charts using Streamlit Native Bar Chart
    st.bar_chart(sev_df.set_index("Severity"), color="#818cf8", height=200)

# Lower Section: Timeline & Logs Table
st.markdown("---")
col_timeline, col_logs = st.columns([1, 1])

with col_timeline:
    st.markdown("### 📈 Detection Timeline (Past 24 Hours)")
    if stats["hourly"]:
        hourly_df = pd.DataFrame({
            "Hour": [f"{h}:00" for h in stats["hourly"].keys()],
            "Detections": list(stats["hourly"].values())
        })
        st.line_chart(hourly_df.set_index("Hour"), color="#a855f7", height=260)
    else:
        st.info("No logs in the past 24 hours.")

with col_logs:
    st.markdown("### 📝 Recent Pothole Records")
    recent_list = stats["recent"]
    
    if not recent_list:
        st.info("Waiting for first pothole detection from live stream...")
    else:
        # Create a display list of logs
        log_records = []
        for log in recent_list:
            # Parse ISO timestamp
            dt = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
            local_time = dt.strftime("%H:%M:%S")
            log_records.append({
                "Record ID": log["id"],
                "Time": local_time,
                "Confidence": f"{int(log['confidence'] * 100)}%",
                "Severity": log["severity"],
                "Image Path": log["image_path"]
            })
            
        logs_df = pd.DataFrame(log_records)
        
        # Display the logs table (excluding Image Path for display size)
        st.dataframe(logs_df[["Time", "Confidence", "Severity"]], width="stretch", hide_index=True)

        # Snapshot Preview functionality
        st.markdown("#### 🔍 Snapshot Inspector")
        # Let user choose a record ID to preview
        selected_record = st.selectbox(
            "Select detection to preview snapshot:",
            options=[r["Record ID"] for r in log_records],
            format_func=lambda x: f"Record #{x} - {next(item for item in log_records if item['Record ID'] == x)['Severity']} Severity"
        )
        
        if selected_record:
            item = next(i for i in log_records if i["Record ID"] == selected_record)
            img_url = f"{API_URL}{item['Image Path']}"
            st.image(img_url, caption=f"Pothole Snapshot (Record #{item['Record ID']} - Detected at {item['Time']})", width="stretch")

# Handle live autorefresh mechanism
if live_reload:
    time.sleep(3.0)
    st.rerun()
