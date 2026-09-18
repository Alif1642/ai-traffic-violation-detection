from __future__ import annotations

import hashlib
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="AI Traffic Violation Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
VEHICLE_MODEL_PATH = MODEL_DIR / "yolov8n.pt"
PLATE_MODEL_PATH = MODEL_DIR / "license_plate_detector.pt"
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "sample_video.mp4"

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
CLASS_COUNTS = {name: 0 for name in VEHICLE_CLASSES.values()}
MAX_UPLOAD_MB = 50


@st.cache_resource(show_spinner=False)
def load_vehicle_model() -> YOLO:
    return YOLO(str(VEHICLE_MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_plate_model() -> YOLO:
    return YOLO(str(PLATE_MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_ocr() -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def session_directory() -> Path:
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / "traffic-demo" / st.session_state.session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    destination = session_directory() / f"input{suffix}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def get_video_details(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The selected video could not be opened.")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def clamp_box(box: tuple[int, int, int, int], width: int, height: int):
    x1, y1, x2, y2 = box
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(1, min(x2, width)),
        max(1, min(y2, height)),
    )


def read_plate(vehicle_crop: np.ndarray, plate_model: YOLO, ocr) -> str:
    best_text = ""
    results = plate_model.predict(vehicle_crop, conf=0.35, imgsz=640, verbose=False)
    crop_height, crop_width = vehicle_crop.shape[:2]

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            coords = tuple(map(int, box.xyxy[0].tolist()))
            px1, py1, px2, py2 = clamp_box(coords, crop_width, crop_height)
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            words = ocr.readtext(plate_crop, detail=0, paragraph=False)
            candidate = " ".join(str(word).strip() for word in words).strip()
            if len(candidate) > len(best_text):
                best_text = candidate

    return best_text or "Unknown"


def make_browser_video(source: Path) -> Path:
    converted = source.with_name("output_browser.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(converted),
    ]
    try:
        subprocess.run(command, check=True, timeout=180)
        return converted if converted.exists() else source
    except (subprocess.SubprocessError, OSError):
        return source


def process_video(
    video_path: Path,
    confidence: float,
    frame_skip: int,
    max_processed_frames: int,
    show_boxes: bool,
    save_output: bool,
    enable_ocr: bool,
    metric_cards,
):
    with st.spinner("Loading AI models..."):
        vehicle_model = load_vehicle_model()
        plate_model = load_plate_model() if enable_ocr else None
        ocr = load_ocr() if enable_ocr else None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        st.error("The video could not be opened. Try an MP4/H.264 file.")
        return

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(
        max_processed_frames,
        max(1, int(np.ceil(total_source_frames / frame_skip))),
    )

    output_path = session_directory() / "output_raw.mp4"
    writer = None
    if save_output:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, source_fps / frame_skip),
            (width, height),
        )
        if not writer.isOpened():
            writer = None
            st.warning("Output video writer could not start; analysis will continue.")

    preview = st.empty()
    progress = st.progress(0.0)
    status = st.empty()

    counts = CLASS_COUNTS.copy()
    tracked_ids: set[int] = set()
    previous_position: dict[int, tuple[int, int]] = {}
    detections: list[dict[str, object]] = []
    screenshot_paths: list[Path] = []
    source_frame = 0
    processed_frame = 0
    road_center_x = width // 2

    try:
        while capture.isOpened() and processed_frame < max_processed_frames:
            ok, frame = capture.read()
            if not ok:
                break

            source_frame += 1
            if source_frame % frame_skip != 0:
                continue

            processed_frame += 1
            annotated = frame.copy()
            results = vehicle_model.track(
                frame,
                persist=True,
                classes=list(VEHICLE_CLASSES),
                conf=confidence,
                imgsz=640,
                verbose=False,
            )

            result = results[0] if results else None
            boxes = result.boxes if result is not None else None

            if boxes is not None:
                for box in boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    class_id = int(box.cls.item())
                    if class_id not in VEHICLE_CLASSES:
                        continue

                    confidence_score = float(box.conf.item())
                    vehicle = VEHICLE_CLASSES[class_id]
                    coords = tuple(map(int, box.xyxy[0].tolist()))
                    x1, y1, x2, y2 = clamp_box(coords, width, height)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    lane = "Left" if center_x < road_center_x else "Right"
                    motion_score = 0.0
                    if track_id in previous_position:
                        old_x, old_y = previous_position[track_id]
                        motion_score = float(np.hypot(center_x - old_x, center_y - old_y))
                    previous_position[track_id] = (center_x, center_y)

                    violations = []
                    if vehicle in {"Bus", "Truck"} and lane == "Right":
                        violations.append("Possible wrong lane")
                    violation = ", ".join(violations) if violations else "No rule triggered"

                    plate_text = "Not checked"
                    vehicle_crop = frame[y1:y2, x1:x2]

                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        counts[vehicle] += 1

                        if enable_ocr and plate_model is not None and ocr is not None:
                            plate_text = read_plate(vehicle_crop, plate_model, ocr)

                        screenshot_path = session_directory() / f"vehicle_{track_id}.jpg"
                        cv2.imwrite(str(screenshot_path), vehicle_crop)
                        screenshot_paths.append(screenshot_path)

                        detections.append(
                            {
                                "ID": track_id,
                                "Vehicle": vehicle,
                                "Lane": lane,
                                "Plate": plate_text,
                                "Rule result": violation,
                                "Motion score (px)": round(motion_score, 2),
                                "Confidence": round(confidence_score, 4),
                            }
                        )

                    if show_boxes:
                        color = (0, 180, 0) if not violations else (0, 0, 255)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{vehicle} | ID {track_id} | {violation}"
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

            if writer is not None:
                writer.write(annotated)

            if processed_frame == 1 or processed_frame % 3 == 0:
                preview.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

            for card, label, key in zip(
                metric_cards,
                ["🚗 Cars", "🏍 Motorcycles", "🚌 Buses", "🚚 Trucks"],
                ["Car", "Motorcycle", "Bus", "Truck"],
            ):
                card.metric(label, counts[key])

            completion = min(processed_frame / target_frames, 1.0)
            progress.progress(completion)
            status.info(f"Processed {processed_frame} of up to {target_frames} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    progress.empty()
    status.success("Detection completed.")

    total_vehicles = sum(counts.values())
    total_flags = sum(row["Rule result"] != "No rule triggered" for row in detections)
    st.subheader("📊 Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique vehicles", total_vehicles)
    col2.metric("Rule flags", total_flags)
    col3.metric("Processed frames", processed_frame)

    if detections:
        dataframe = pd.DataFrame(detections)
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download results (CSV)",
            dataframe.to_csv(index=False).encode("utf-8"),
            "vehicle_detection_results.csv",
            "text/csv",
        )
    else:
        st.warning("No supported vehicle class was detected.")

    if writer is not None and output_path.exists():
        browser_video = make_browser_video(output_path)
        st.subheader("🎥 Processed video")
        st.video(str(browser_video))
        st.download_button(
            "⬇ Download processed video",
            browser_video.read_bytes(),
            "traffic_detection_output.mp4",
            "video/mp4",
        )

    if screenshot_paths:
        with st.expander("📸 Vehicle screenshots"):
            columns = st.columns(4)
            for index, image_path in enumerate(screenshot_paths[:20]):
                columns[index % 4].image(str(image_path), caption=image_path.stem)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    .stButton > button {width: 100%; min-height: 3rem; font-weight: 700;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: .8rem; border-radius: .75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚦 AI Traffic Violation Detection")
st.caption("YOLOv8 • EasyOCR • OpenCV • Streamlit")
st.info(
    "Academic demonstration only. Lane flags and motion scores are heuristic and "
    "must not be used for real-world enforcement or legal decisions."
)

missing_models = [
    path.name for path in (VEHICLE_MODEL_PATH, PLATE_MODEL_PATH) if not path.exists()
]
if missing_models:
    st.error("Missing model file(s): " + ", ".join(missing_models))
    st.stop()

st.sidebar.header("⚙️ Detection settings")
confidence = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 8, 3)
max_processed_frames = st.sidebar.slider("Maximum processed frames", 30, 600, 180, 30)
enable_ocr = st.sidebar.checkbox("Enable plate OCR (slower)", value=False)
show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
save_output = st.sidebar.checkbox("Create downloadable video", value=True)

metric_columns = st.columns(4)
metric_cards = [column.empty() for column in metric_columns]
for card, label in zip(
    metric_cards,
    ["🚗 Cars", "🏍 Motorcycles", "🚌 Buses", "🚚 Trucks"],
):
    card.metric(label, 0)

st.subheader("1. Choose a video")
source_options = ["Upload a video"]
if SAMPLE_VIDEO_PATH.exists():
    source_options.insert(0, "Use sample video")
source = st.radio("Video source", source_options, horizontal=True, label_visibility="collapsed")

video_path = None
if source == "Use sample video":
    video_path = SAMPLE_VIDEO_PATH
else:
    uploaded_video = st.file_uploader(
        "Upload an MP4, MOV, AVI or MKV file (maximum 50 MB)",
        type=["mp4", "mov", "avi", "mkv"],
    )
    if uploaded_video is not None:
        size_mb = uploaded_video.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"The uploaded file is {size_mb:.1f} MB. Maximum size is {MAX_UPLOAD_MB} MB.")
        else:
            video_path = save_uploaded_video(uploaded_video)

if video_path is not None:
    try:
        details = get_video_details(video_path)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.video(str(video_path))
    info_columns = st.columns(4)
    info_columns[0].metric("Frames", details["frames"])
    info_columns[1].metric("FPS", f"{details['fps']:.2f}")
    info_columns[2].metric("Duration", f"{details['duration']:.1f} sec")
    info_columns[3].metric("Resolution", f"{details['width']} × {details['height']}")

    st.subheader("2. Run detection")
    if st.button("🚀 Start detection", type="primary"):
        process_video(
            video_path=video_path,
            confidence=confidence,
            frame_skip=frame_skip,
            max_processed_frames=max_processed_frames,
            show_boxes=show_boxes,
            save_output=save_output,
            enable_ocr=enable_ocr,
            metric_cards=metric_cards,
        )
else:
    st.caption("Upload a short video to enable detection.")

st.divider()
st.caption("Developed as a final-year academic project.")
