import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

import torch

AI_MODEL_PATH = Path(__file__).parents[1] / "ai model"
sys.path.insert(0, str(AI_MODEL_PATH))
from labels import CHEXPERT_LABELS
from train_chexpert import CheXpertDataset, build_model

MODULE_PATH = Path(__file__).parents[1] / "ai model" / "triage.py"
spec = importlib.util.spec_from_file_location("triage", MODULE_PATH)
triage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = triage
spec.loader.exec_module(triage)

backend_path = Path(__file__).parents[1] / "backend" / "app.py"
backend_spec = importlib.util.spec_from_file_location("backend_app", backend_path)
backend_app = importlib.util.module_from_spec(backend_spec)
sys.modules[backend_spec.name] = backend_app
backend_spec.loader.exec_module(backend_app)


class TriageTests(unittest.TestCase):
    def test_high_priority_combines_severity_and_confidence(self):
        result = triage.triage_prediction(
            {"Pneumonia": 0.95, "No Finding": 0.05},
            {"Pneumonia": 1.0, "No Finding": 0.0},
        )
        self.assertEqual(result.priority, "High")
        self.assertAlmostEqual(result.risk, 0.715)

    def test_low_priority_is_possible_for_no_finding(self):
        result = triage.triage_prediction(
            {"No Finding": 0.99, "Pneumonia": 0.01},
            {"No Finding": 0.0, "Pneumonia": 1.0},
        )
        self.assertEqual(result.priority, "Low")
        self.assertEqual(result.finding, "No Finding")

    def test_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            triage.triage_prediction({"Pneumonia": 1.2}, {"Pneumonia": 1.0})

    def test_demo_checkpoint_can_show_multiple_priorities(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "demo.pt"
            backend_app.ensure_demo_checkpoint(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue(payload["demo"])

            model = backend_app.PneumoniaInference(checkpoint)
            normal_sample = Image.new("RGB", (224, 224), color="white")
            ImageDraw.Draw(normal_sample).ellipse((50, 50, 174, 174), outline="black", width=5)
            dark_sample = Image.new("RGB", (224, 224), color="black")
            normal_result = model.predict(normal_sample)
            high_result = model.predict(dark_sample)
            self.assertGreater(normal_result["probabilities"]["No Finding"], normal_result["probabilities"]["Pneumonia"])
            self.assertEqual(normal_result["triage"]["priority"], "Low")
            self.assertEqual(normal_result["mode"], "demo")
            self.assertEqual(normal_result["heatmap_kind"], "image_contrast_demo")
            self.assertGreater(max(max(row) for row in normal_result["heatmap"]), 0.0)
            self.assertGreater(high_result["probabilities"]["Pneumonia"], 0.7)
            self.assertEqual(high_result["triage"]["priority"], "High")

    def test_chexpert_dataset_handles_uncertain_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            Image.new("RGB", (32, 32), color="white").save(root / "sample.png")
            row = {"Path": "sample.png", **{label: 0.0 for label in CHEXPERT_LABELS}}
            row["Edema"] = -1.0
            row["Pneumonia"] = float("nan")
            manifest = root / "labels.csv"
            pd.DataFrame([row]).to_csv(manifest, index=False)

            _, zero_labels, zero_mask = CheXpertDataset(manifest, root, uncertain_label_policy="zeros")[0]
            _, ignore_labels, ignore_mask = CheXpertDataset(manifest, root, uncertain_label_policy="ignore")[0]
            _, one_labels, one_mask = CheXpertDataset(manifest, root, uncertain_label_policy="ones")[0]

            edema_index = CHEXPERT_LABELS.index("Edema")
            pneumonia_index = CHEXPERT_LABELS.index("Pneumonia")
            self.assertEqual(zero_labels[edema_index].item(), 0.0)
            self.assertEqual(zero_mask[edema_index].item(), 1.0)
            self.assertEqual(ignore_mask[edema_index].item(), 0.0)
            self.assertEqual(one_labels[edema_index].item(), 1.0)
            self.assertEqual(one_mask[edema_index].item(), 1.0)
            self.assertEqual(zero_mask[pneumonia_index].item(), 0.0)

    def test_chexpert_checkpoint_inference_returns_all_14_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "chexpert.pt"
            model = build_model()
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.zero_()
                model.classifier.bias[CHEXPERT_LABELS.index("Edema")] = 4.0
            torch.save({"model": model.state_dict(), "labels": CHEXPERT_LABELS}, checkpoint)

            engine = backend_app.CheXpertInference(checkpoint, device="cpu")
            result = engine.predict(Image.new("RGB", (224, 224), color="white"))

            self.assertEqual(result["model_type"], "chexpert_multilabel")
            self.assertEqual(set(result["probabilities"]), set(CHEXPERT_LABELS))
            self.assertEqual(result["triage"]["finding"], "Edema")
            self.assertEqual(result["heatmap_kind"], "grad_cam")


if __name__ == "__main__":
    unittest.main()
