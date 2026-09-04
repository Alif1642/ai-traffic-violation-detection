# Local setup

## Prerequisites

- Python 3.10-3.12 (Python 3.11 recommended)
- Git
- Sufficient RAM for YOLO and EasyOCR

## Windows PowerShell

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd ai-traffic-violation-detection

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, upload an authorized traffic video, choose the settings, and select **Start Detection**.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m compileall -q app.py tests
ruff check app.py tests
pytest -q
```

## Troubleshooting

- Confirm both `.pt` files exist under `models/`.
- Use the repository root as the working directory.
- The first EasyOCR initialization may take longer than later runs.
- Long or high-resolution videos can require substantial CPU, memory, and processing time.
- If output video playback has codec limitations, download it and open it in a desktop media player.

