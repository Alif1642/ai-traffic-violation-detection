from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
MODEL_DIR = PROJECT_ROOT / "models"

for directory in (UPLOAD_DIR, OUTPUT_DIR, SCREENSHOT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_vehicle_model():
    return YOLO(str(MODEL_DIR / "yolov8n.pt"))


@st.cache_resource
def load_plate_model():
    return YOLO(str(MODEL_DIR / "license_plate_detector.pt"))


@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

vehicle_model = load_vehicle_model()
plate_model = load_plate_model()
ocr = load_ocr()

# =====================================================
# VEHICLE CLASSES
# =====================================================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{

    width:100%;
    height:55px;

    background:#00FF99;
    color:black;

    border-radius:10px;

    font-size:18px;

    font-weight:bold;

}

div[data-testid="metric-container"]{

    background:#1E222A;

    border-radius:12px;

    padding:15px;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🚦 AI Traffic Violation Detection System")

st.write(
    "Upload CCTV Road Video to Detect Vehicles, Number Plates and Traffic Violations."
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence",
    0.10,
    1.00,
    0.40,
    0.05
)

show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes",
    True
)

save_output = st.sidebar.checkbox(
    "Save Output Video",
    True
)

# =====================================================
# LIVE DASHBOARD
# =====================================================

c1, c2, c3, c4 = st.columns(4)

car_card = c1.empty()
bike_card = c2.empty()
bus_card = c3.empty()
truck_card = c4.empty()

car_card.metric("🚗 Cars", 0)
bike_card.metric("🏍 Bikes", 0)
bus_card.metric("🚌 Bus", 0)
truck_card.metric("🚚 Truck", 0)

st.divider()

# =====================================================
# VIDEO UPLOAD
# =====================================================

uploaded_video = st.file_uploader(

    "📤 Upload CCTV Road Video",

    type=["mp4", "avi", "mov", "mkv"]

)

# =====================================================
# IF VIDEO UPLOADED
# =====================================================

if uploaded_video is not None:

    safe_filename = Path(uploaded_video.name).name
    video_path = UPLOAD_DIR / safe_filename

    with video_path.open("wb") as f:

        f.write(uploaded_video.read())

    st.success("✅ Video Uploaded Successfully")

    st.video(video_path)

    # -------------------------------------
    # Video Information
    # -------------------------------------

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = 0

    if fps > 0:

        duration = total_frames / fps

    cap.release()

    st.subheader("📹 Video Information")

    a, b, c, d = st.columns(4)

    a.metric("Frames", total_frames)

    b.metric("FPS", round(fps, 2))

    c.metric(
        "Duration",
        f"{round(duration,2)} sec"
    )

    d.metric(
        "Resolution",
        f"{width} x {height}"
    )

    st.divider()

    start = st.button(
        "🚀 Start Detection"
    )
    # =====================================================
# START DETECTION
# =====================================================

    if start:

        cap = cv2.VideoCapture(video_path)

        # Output Video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        output_path = OUTPUT_DIR / "output.mp4"
        writer_fps = fps if fps > 0 else 30.0
        out = None

        if save_output:
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                writer_fps,
                (width, height)
            )

        # Road Center
        ROAD_CENTER_X = width // 2

        # Live Preview
        preview = st.empty()

        progress = st.progress(0)

        status = st.empty()

        # Databases
        tracked_ids = set()

        previous_position = {}

        detection_results = []

        screenshot_count = 0

        car_count = 0
        bike_count = 0
        bus_count = 0
        truck_count = 0
                # =====================================================
        # MAIN LOOP
        # =====================================================
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = vehicle_model.track(

                frame,

                persist=True,

                conf=confidence,

                verbose=False

            )

            annotated = frame.copy()
                        # ==========================================
            # VEHICLE DETECTION
            # ==========================================

            if len(results) == 0:
                continue

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                if box.id is None:
                    continue

                track_id = int(box.id.item())

                cls = int(box.cls.item())

                conf = float(box.conf.item())

                if cls not in VEHICLE_CLASSES:
                    continue

                vehicle = VEHICLE_CLASSES[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Vehicle Center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Lane
                if center_x < ROAD_CENTER_X:
                    lane = "Left"
                else:
                    lane = "Right"

                # Crop Vehicle
                vehicle_crop = frame[y1:y2, x1:x2]

                if vehicle_crop.size == 0:
                    continue
                                # ==========================================
                # SPEED ESTIMATION
                # ==========================================

                if track_id in previous_position:

                    old_x, old_y = previous_position[track_id]

                    distance = np.sqrt(
                        (center_x - old_x) ** 2 +
                        (center_y - old_y) ** 2
                    )

                    speed = distance * 0.45

                else:

                    speed = 0

                previous_position[track_id] = (
                    center_x,
                    center_y
                )

                # ==========================================
                # VIOLATION DETECTION
                # ==========================================

                violation = "Safe"

                # Over Speed

                if speed > 60:

                    violation = "Over Speed"

                # Wrong Lane
                # Bus & Truck should stay Left

                if vehicle in ["Bus", "Truck"] and lane == "Right":
                    violation = "Wrong Lane"
                                        # ==========================================
                # LICENSE PLATE DETECTION
                # ==========================================

                plate_text = ""

                plate_results = plate_model.predict(

                    vehicle_crop,

                    conf=0.35,

                    verbose=False

                )

                for plate in plate_results:

                    if plate.boxes is None:
                        continue

                    for pbox in plate.boxes:

                        px1, py1, px2, py2 = map(
                            int,
                            pbox.xyxy[0]
                        )

                        plate_crop = vehicle_crop[
                            py1:py2,
                            px1:px2
                        ]

                        if plate_crop.size == 0:
                            continue

                        ocr_result = ocr.readtext(plate_crop)
                        plate_text = ""

                        if ocr_result:

                            for item in ocr_result:

                                plate_text += item[1] + " "

                        plate_text = plate_text.strip()

                if plate_text == "":
                    plate_text = "Unknown"
                                    # ==========================================
                # SAVE UNIQUE DETECTION
                # ==========================================

                if track_id not in tracked_ids:

                    tracked_ids.add(track_id)

                    # Vehicle Count

                    if vehicle == "Car":
                        car_count += 1

                    elif vehicle == "Motorcycle":
                        bike_count += 1

                    elif vehicle == "Bus":
                        bus_count += 1

                    elif vehicle == "Truck":
                        truck_count += 1

                    # Save Screenshot

                    screenshot_name = SCREENSHOT_DIR / f"{track_id}.jpg"

                    cv2.imwrite(
                        str(screenshot_name),
                        vehicle_crop
                    )

                    screenshot_count += 1

                    # Save Result

                    detection_results.append({

                        "ID": track_id,

                        "Vehicle": vehicle,

                        "Lane": lane,

                        "Plate Number": plate_text,

                        "Violation": violation,

                        "Confidence": f"{conf*100:.2f}%"

                    })
                                    # ==========================================
                # DRAW RESULT ON VIDEO
                # ==========================================

                color = (0,255,0)

                if violation != "Safe":

                    color = (0,0,255)

                if show_boxes:
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    label = f"{vehicle} | {plate_text}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        violation,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                        # ==========================================
            # LIVE DASHBOARD UPDATE
            # ==========================================

            car_card.metric("🚗 Cars", car_count)
            bike_card.metric("🏍 Bikes", bike_count)
            bus_card.metric("🚌 Bus", bus_count)
            truck_card.metric("🚚 Truck", truck_count)

            # ==========================================
            # WRITE OUTPUT VIDEO
            # ==========================================

            if out is not None:
                out.write(annotated)

            preview.image(

                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),

                channels="RGB",

                use_container_width=True

            )

            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_total > 0:

                progress.progress(

                    min(current / frame_total, 1.0)

                )

            status.info(

                f"Processing Frame : {current}/{frame_total}"

            )
                # ==========================================
        # RELEASE
        # ==========================================

        cap.release()

        if out is not None:
            out.release()

        progress.empty()

        status.success("✅ Detection Completed")
                # ==========================================
        # OUTPUT VIDEO
        # ==========================================

        st.success("🎉 Vehicle Detection Completed")

        if save_output and output_path.exists():
            st.subheader("🎥 Output Video")
            st.video(str(output_path))

            with output_path.open("rb") as file:
                st.download_button(
                    "⬇ Download Output Video",
                    data=file,
                    file_name="output.mp4",
                    mime="video/mp4",
                )
                # ==========================================
        # STATISTICS
        # ==========================================

        total_vehicles = (

            car_count +

            bike_count +

            bus_count +

            truck_count

        )

        total_violations = sum(

            1 for row in detection_results

            if row["Violation"] != "Safe"

        )

        st.success(f"📸 Screenshots Saved : {screenshot_count}")

        st.subheader("📊 Statistics")

        a,b,c = st.columns(3)

        a.metric("🚘 Total Vehicles", total_vehicles)

        b.metric("⚠ Total Violations", total_violations)

        if total_vehicles > 0:

            percent = round(total_violations*100/total_vehicles,2)

        else:

            percent = 0

        c.metric("Violation Rate", f"{percent}%")
                # ==========================================
        # DETECTION RESULTS
        # ==========================================

        st.subheader("📋 Detection Results")

        if len(detection_results) > 0:

            df = pd.DataFrame(detection_results)

            st.dataframe(

                df,

                use_container_width=True

            )

            # CSV Download

            csv = df.to_csv(

                index=False

            ).encode("utf-8")

            st.download_button(

                label="⬇ Download Detection Results (CSV)",

                data=csv,

                file_name="vehicle_detection_results.csv",

                mime="text/csv"

            )

        else:

            st.warning("No Vehicle Detected")
        
                # ==========================================
        # SCREENSHOTS
        # ==========================================

        st.subheader("📸 Saved Screenshots")

        image_files = sorted(SCREENSHOT_DIR.glob("*.jpg"))

        if len(image_files) > 0:

            cols = st.columns(4)

            for i, img in enumerate(image_files):

                cols[i % 4].image(

                    str(img),

                    caption=img.name,

                    use_container_width=True

                )

        else:

            st.info("No Screenshot Saved")
            # ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
"""
<center>

### 🚦 AI Traffic Violation Detection System

YOLOv8 • PaddleOCR • OpenCV • Streamlit

Developed for Final Year Project

</center>
""",
unsafe_allow_html=True
)

        
