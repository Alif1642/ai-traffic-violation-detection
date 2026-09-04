import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_entrypoint_is_valid_python() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_required_model_files_exist() -> None:
    assert (ROOT / "models" / "yolov8n.pt").is_file()
    assert (ROOT / "models" / "license_plate_detector.pt").is_file()


def test_app_has_no_original_windows_model_path() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "D:/Vio" not in source
