# AI Imaging Triage

A research prototype for chest X-ray analysis, with a FastAPI backend and a browser-based frontend. It can use a CheXpert 14-observation checkpoint when one is available.

## Run locally on Windows

Use Python 3.14 and PowerShell from the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In one terminal, start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```powershell
.\.venv\Scripts\python.exe -m http.server 5173 --bind 127.0.0.1 --directory frontend
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Models and datasets

This repository intentionally excludes X-ray datasets, dataset manifests, trained checkpoints, and the research-paper PDF. Obtain datasets and model weights only from sources whose terms permit your intended use. Place a permitted CheXpert checkpoint at `ai model/checkpoints/chexpert_densenet121.pt` to enable 14-observation inference.

Without a trained checkpoint, the application creates a **demo-only** fallback. It is not a disease prediction. The optional report endpoint expects a separately running local Ollama service; otherwise it returns a structured fallback report.

See [CheXpert training and Cloud Run notes](documentation/chexpert_multilabel_workflow.md) and [pneumonia training notes](documentation/pneumonia_cloud_workflow.md).

## Safety

This is research software, not a medical device or diagnostic service. Do not upload identifiable patient images. Outputs require qualified clinical review and must not be used as standalone medical advice.
