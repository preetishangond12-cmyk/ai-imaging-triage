"""FastAPI service for CheXpert analysis and risk-aware triage."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
import urllib.error
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
AI_MODEL = ROOT / "ai model"
sys.path.insert(0, str(AI_MODEL))

PneumoniaInference = import_module("inference").PneumoniaInference
CheXpertInference = import_module("inference").CheXpertInference


class InferenceEngine(Protocol):
    def predict(self, image: Image.Image) -> dict[str, Any]: ...


class ReportRequest(BaseModel):
    finding: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1)
    severity: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    priority: str = Field(pattern="^(High|Medium|Low)$")
    mode: str = Field(pattern="^(demo|trained_model)$")
    probabilities: dict[str, float] = Field(default_factory=dict, max_length=20)


app = FastAPI(title="Edge-AI Medical Image Triage", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
CHECKPOINT = ROOT / "ai model" / "checkpoints" / "pneumonia_densenet121.pt"
CHEXPERT_CHECKPOINT = ROOT / "ai model" / "checkpoints" / "chexpert_densenet121.pt"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
inference_engine: InferenceEngine | None = None


def ensure_demo_checkpoint(path: Path) -> None:
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict) and payload.get("demo") is True:
                return
        except Exception:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([4.0, -4.0]))
    payload = {"model": model.state_dict(), "labels": ["No Finding", "Pneumonia"], "demo": True}
    torch.save(payload, path)


def checkpoint_is_demo(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("demo") is True


def get_engine() -> InferenceEngine:
    global inference_engine
    if inference_engine is None:
        if CHEXPERT_CHECKPOINT.is_file():
            inference_engine = CheXpertInference(CHEXPERT_CHECKPOINT)
        else:
            ensure_demo_checkpoint(CHECKPOINT)
            inference_engine = PneumoniaInference(CHECKPOINT)
    return inference_engine


@app.get("/api/health")
def health() -> dict[str, object]:
    if CHEXPERT_CHECKPOINT.is_file():
        return {
            "status": "ready",
            "checkpoint": str(CHEXPERT_CHECKPOINT),
            "model_available": True,
            "model_type": "chexpert_multilabel",
            "labels": import_module("labels").CHEXPERT_LABELS,
        }
    ensure_demo_checkpoint(CHECKPOINT)
    demo_mode = checkpoint_is_demo(CHECKPOINT)
    return {
        "status": "demo_mode" if demo_mode else "ready" if CHECKPOINT.exists() else "model_missing",
        "checkpoint": str(CHECKPOINT),
        "model_available": CHECKPOINT.exists() and not demo_mode,
        "model_type": "pneumonia_binary",
        "labels": ["No Finding", "Pneumonia"],
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or other image file.")
    payload = await file.read()
    if len(payload) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 15 MB.")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="The uploaded image could not be read.") from error

    result = get_engine().predict(image)
    result["filename"] = file.filename or "uploaded-image"
    result["disclaimer"] = (
        "Demo output only: this checkpoint is not trained to recognize disease. The heatmap shows image contrast, not pathology."
        if result.get("mode") == "demo"
        else "Research support only. Predictions and prototype risk values are not diagnoses; a qualified clinician must review this result."
    )
    return result


@app.post("/api/report")
def generate_report(request: ReportRequest) -> dict[str, str]:
    """Ask a local LLM for a review-oriented report without changing the triage result."""
    prompt = f"""You are drafting a technical review note for a student medical AI prototype.
Do not diagnose, recommend treatment, or claim clinical validity.
Describe only the supplied values and identify what a qualified clinician should review.
Use 3 short sections: Summary, Interpretation, Review Needed.

Mode: {request.mode}
Finding: {request.finding}
Confidence: {request.confidence:.3f}
Severity: {request.severity:.3f}
Risk: {request.risk:.3f}
Priority: {request.priority}
Probabilities: {json.dumps(request.probabilities, sort_keys=True)}
"""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Be concise, factual, cautious, and clearly non-diagnostic."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }).encode("utf-8")
    ollama_request = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(ollama_request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        report = result.get("message", {}).get("content", "")
        if not report:
            raise RuntimeError("Ollama returned an empty response")
        return {"report": report, "model": OLLAMA_MODEL, "provider": "local_ollama"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
        mode_note = (
            "This is demo output from an untrained checkpoint, not a medical prediction."
            if request.mode == "demo"
            else "This is a research-model output and requires qualified clinical review."
        )
        top_finding = max(request.probabilities, key=request.probabilities.get) if request.probabilities else request.finding
        fallback_report = (
            "Summary\n"
            f"The structured result identifies {request.finding} with "
            f"{request.confidence:.1%} confidence and assigns {request.priority.lower()} priority.\n\n"
            "Interpretation\n"
            f"The calculated risk score is {request.risk:.3f} from severity {request.severity:.3f} "
            "and confidence data. The probability field named "
            f"{top_finding} is the highest supplied class value, if probabilities were provided.\n\n"
            "Review Needed\n"
            f"{mode_note} A qualified clinician should verify the image, original prediction, explanation map, and clinical context."
        )
        return {
            "report": fallback_report,
            "model": "structured_fallback",
            "provider": "local_rules",
            "llm_error": str(error),
        }


@app.get("/api/llm-health")
def llm_health() -> dict[str, object]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [item["name"] for item in payload.get("models", [])]
        return {"available": True, "model": OLLAMA_MODEL, "model_installed": OLLAMA_MODEL in models}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {"available": False, "model": OLLAMA_MODEL, "model_installed": False}


@app.get("/api/heatmap-preview")
def heatmap_preview() -> dict[str, str]:
    """Small endpoint contract check for clients; analysis returns the actual map."""
    pixel = Image.new("RGB", (1, 1), (16, 31, 42))
    buffer = io.BytesIO()
    pixel.save(buffer, format="PNG")
    return {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
