"""Checkpoint-backed chest X-ray inference and Grad-CAM explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from PIL import ImageFilter

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from labels import CHEXPERT_LABELS, CHEXPERT_SEVERITY
from triage import triage_prediction

LABELS = ["No Finding", "Pneumonia"]
SEVERITY = {"No Finding": 0.0, "Pneumonia": 0.9}


class PneumoniaInference:
    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.is_demo = self._checkpoint_is_demo(self.checkpoint)
        self.model = self._load_model(self.checkpoint).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.model.features.register_forward_hook(self._capture_activations)

    @staticmethod
    def _checkpoint_is_demo(checkpoint: Path) -> bool:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        return isinstance(payload, dict) and payload.get("demo") is True

    @staticmethod
    def _load_model(checkpoint: Path) -> torch.nn.Module:
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")

        if isinstance(payload, dict) and "model" in payload:
            state_dict = payload["model"]
            if payload.get("demo") is True:
                with torch.no_grad():
                    model.classifier.weight.zero_()
                    model.classifier.bias.copy_(torch.tensor([4.0, -4.0]))
                return model
        else:
            state_dict = payload

        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint at {checkpoint} must contain a model state dict.")

        model.load_state_dict(state_dict)
        return model

    def _capture_activations(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        self.activations = output
        output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("L")).repeat(1, 3, 1, 1).to(self.device)
        tensor.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        model_probabilities = torch.softmax(logits, dim=1)[0]

        if self.is_demo:
            grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            darkness = float(np.clip(1.0 - grayscale.mean(), 0.0, 1.0))
            contrast = float(np.clip(grayscale.std() * 1.8, 0.0, 1.0))
            pneumonia_probability = float(np.clip(0.18 + 0.57 * darkness + 0.18 * contrast, 0.05, 0.93))
            probabilities = torch.tensor([1.0 - pneumonia_probability, pneumonia_probability], device=model_probabilities.device)
            resized = image.convert("L").resize((224, 224))
            local_average = np.asarray(resized.filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.float32)
            detail = np.abs(np.asarray(resized, dtype=np.float32) - local_average)
            scale = float(np.percentile(detail, 99))
            heatmap = np.clip(detail / max(scale, 1.0), 0.0, 1.0)
        else:
            probabilities = model_probabilities
            predicted_index = int(probabilities.argmax())
            logits[0, predicted_index].backward()
            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture model features")
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            heatmap = cam[0, 0].detach().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
        named_probabilities = {
            label: float(value)
            for label, value in zip(LABELS, probabilities.detach().cpu())
        }
        triage = triage_prediction(named_probabilities, SEVERITY)
        return {
            "probabilities": named_probabilities,
            "triage": triage.to_dict(),
            "heatmap": heatmap.tolist(),
            "mode": "demo" if self.is_demo else "trained_model",
            "model_type": "pneumonia_binary",
            "heatmap_kind": "image_contrast_demo" if self.is_demo else "grad_cam",
        }


class CheXpertInference:
    """Multi-label inference for a trained 14-observation CheXpert checkpoint."""

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise TypeError(f"Checkpoint at {self.checkpoint} must contain model weights and label metadata.")
        if payload.get("labels") != CHEXPERT_LABELS:
            raise ValueError(f"Checkpoint at {self.checkpoint} does not have the expected CheXpert labels.")
        self.model = models.densenet121(weights=None)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(CHEXPERT_LABELS))
        self.model.load_state_dict(payload["model"])
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((int(payload.get("image_size", 224)),) * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.model.features.register_forward_hook(self._capture_activations)

    def _capture_activations(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        self.activations = output
        output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("L")).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
        tensor.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits)[0]
        predicted_index = int(probabilities.argmax())
        logits[0, predicted_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model features")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        heatmap = cam[0, 0].detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
        named_probabilities = {
            label: float(value)
            for label, value in zip(CHEXPERT_LABELS, probabilities.detach().cpu())
        }
        triage = triage_prediction(named_probabilities, CHEXPERT_SEVERITY)
        return {
            "probabilities": named_probabilities,
            "triage": triage.to_dict(),
            "heatmap": heatmap.tolist(),
            "mode": "trained_model",
            "model_type": "chexpert_multilabel",
            "heatmap_kind": "grad_cam",
        }
