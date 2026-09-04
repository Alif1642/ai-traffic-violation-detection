# Project report and implementation review

## Reported project

The academic report is titled **AI-Based Real-Time Traffic Violation Detection System**. It proposes a low-cost video-based prototype for Bangladesh that detects and tracks vehicles, estimates speed and lane-related violations, attempts license-plate recognition, and records evidence.

## Verified implementation

| Area | Verified state |
|---|---|
| User interface | Streamlit video-upload application |
| Vehicle detection | Ultralytics YOLOv8n COCO classes: car, motorcycle, bus, truck |
| Tracking | `YOLO.track(..., persist=True)` in the Streamlit application; DeepSORT appears in the experimental notebook |
| Plate recognition | Custom YOLO plate detector followed by EasyOCR |
| Speed logic | Inter-frame pixel displacement multiplied by a fixed factor |
| Lane logic | Left/right frame-half assignment; right-side bus/truck rule in the application |
| Outputs | Annotated video, vehicle crops, counts, violation summary, downloadable CSV |
| Storage | Local filesystem only; no database |
| API | None |
| Docker | None |
| Monitoring | Streamlit progress/status only; no production monitoring stack |
| Tests in original files | None |

## Data and preprocessing

The report describes smartphone-captured road video, removal of unsuitable clips, MP4 conversion, frame resizing/extraction, noise reduction, and combining videos. The supplied notebook reads one `data.mp4`, resizes frames to 1280x720, applies Canny/Hough lane-line processing, and runs vehicle detection/tracking. The full raw-data preparation pipeline and original annotation files were not supplied.

## Evaluation evidence and caveat

The supplied research artifacts contain 464 violation rows: 433 `Overspeed` and 31 `Lane Change`. Saved classification reports show approximately 98.9% for Random Forest and SVM and 100% for XGBoost on a 93-row test split.

These scores must not be interpreted as independent violation-detection accuracy:

- the notebook creates `ground_truth.csv` by copying detected violation labels;
- some evaluation cells explicitly assign the same column to `y_true` and `y_pred`;
- the machine-learning models use `Speed` as a feature while violation labels are generated from speed/rule logic, creating strong target leakage;
- the dataset is small and no independent manually labelled holdout video was supplied;
- the 133,000-row evaluation is produced by a many-to-many merge on repeated vehicle/plate values, not 133,000 independently labelled observations.

The repository therefore preserves the reports as experimental artifacts but does not advertise these numbers as validated real-world performance.

## Limitations

- Speed is not calibrated to camera geometry, frame rate, or physical distance.
- Lane interpretation depends on simple frame position/rules and is sensitive to viewpoint and road layout.
- OCR can fail because of blur, resolution, lighting, plate format, and occlusion.
- Detection/tracking can degrade in dense traffic and under occlusion.
- The custom plate model's training details are missing.
- Processing is synchronous and local-file based.

## Evidence-based next steps

1. Create a manually annotated, video-level train/validation/test split.
2. Calibrate camera perspective and validate speed against measured ground truth.
3. Replace the frame-half lane rule with calibrated lane polygons or a lane model.
4. Evaluate detection, tracking, OCR, speed, and violation classification separately.
5. Report class distribution, confidence intervals, inference hardware, and latency/FPS.
6. Add privacy, retention, access-control, and human-review safeguards before real-world use.

