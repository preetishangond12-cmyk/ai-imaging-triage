"""Shared CheXpert observation labels and prototype triage severities."""

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

# Development-only values used by the prototype risk display, not clinical scores.
CHEXPERT_SEVERITY = {
    "Enlarged Cardiomediastinum": 0.6,
    "Cardiomegaly": 0.7,
    "Lung Opacity": 0.6,
    "Lung Lesion": 0.8,
    "Edema": 0.9,
    "Consolidation": 0.8,
    "Pneumonia": 0.9,
    "Atelectasis": 0.6,
    "Pneumothorax": 1.0,
    "Pleural Effusion": 0.7,
    "Pleural Other": 0.5,
    "Fracture": 0.7,
    "Support Devices": 0.0,
    "No Finding": 0.0,
}
