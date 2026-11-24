# Pneumonia Triage Prototype: Cloud Training

## Dataset

Use Kaggle's [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), listed as 5,863 JPEG images in `train`, `val`, and `test` folders, with `NORMAL` and `PNEUMONIA` classes. Kaggle lists a 1.24 GB dataset version and CC BY 4.0 in its dataset description. Review the dataset terms and cite the original dataset/paper in project reports.

This is a binary pneumonia prototype, not the full 14-observation CheXpert experiment. The dataset is from pediatric patients aged one to five, so it does not establish performance for adults or general clinical use. For the 14-finding model, see [chexpert_multilabel_workflow.md](./chexpert_multilabel_workflow.md).

## Keep image data out of the laptop

Train in Google Colab. Only the small training scripts and final model checkpoint need to pass between the laptop and Colab; do not download the dataset ZIP locally.

1. In Kaggle, create an account if needed and accept the dataset terms on the dataset page.
2. In Kaggle Account Settings, create an API token. In Colab, add `KAGGLE_USERNAME` and `KAGGLE_KEY` as notebook secrets and enable access to them. Never paste or share the key in chat or source files.
3. Create a Google Colab notebook and select a GPU runtime if available.
4. Upload `ai model/train_pneumonia.py` and `ai model/evaluate_pneumonia.py` from the project to the Colab session using the file picker. These are small code files, not image data.
5. Run the following cells in Colab.

### Configure Kaggle and fetch the dataset in Colab

```python
from google.colab import userdata
from pathlib import Path
import json
import os

Path('/root/.kaggle').mkdir(exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as token_file:
    json.dump({
        'username': userdata.get('KAGGLE_USERNAME'),
        'key': userdata.get('KAGGLE_KEY'),
    }, token_file)
os.chmod('/root/.kaggle/kaggle.json', 0o600)
```

```python
!pip -q install kaggle
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /content/data --unzip
!find /content/data -maxdepth 4 -type d | head -30
```

Find the folder that contains `train`, `val`, and `test`; it is commonly `/content/data/chest_xray`. Confirm that it contains `train/NORMAL`, `train/PNEUMONIA`, `val/NORMAL`, `val/PNEUMONIA`, `test/NORMAL`, and `test/PNEUMONIA`.

### Train and evaluate

```python
!python /content/train_pneumonia.py \
  --data-dir /content/data/chest_xray \
  --epochs 5 \
  --batch-size 32 \
  --output /content/pneumonia_densenet121.pt
```

```python
!python /content/evaluate_pneumonia.py \
  --data-dir /content/data/chest_xray \
  --checkpoint /content/pneumonia_densenet121.pt
```

If the `find` command showed a different dataset root, substitute that path for `/content/data/chest_xray`. Save the notebook output and record the test metrics; do not claim results until the run has completed.

### Download only the model

```python
from google.colab import files
files.download('/content/pneumonia_densenet121.pt')
```

The downloaded checkpoint is expected to be far smaller than the image archive.

## Run the project on the laptop

Copy the checkpoint into the project at:

```text
ai model/checkpoints/pneumonia_densenet121.pt
```

In a PowerShell terminal opened at the project root:

```powershell
$PYTHON = Join-Path $PWD ".venv\Scripts\python.exe"
& $PYTHON -m uvicorn backend.app:app --reload --port 8000
```

In another PowerShell terminal, also at the project root:

```powershell
$PYTHON = Join-Path $PWD ".venv\Scripts\python.exe"
& $PYTHON -m http.server 5173 --directory frontend
```

Open `http://127.0.0.1:5173`. Upload a chest X-ray JPEG to view the pneumonia/normal probabilities, confidence, Grad-CAM heatmap, risk score, and triage priority.

## Research and safety note

The application is a student research prototype, not a diagnostic product. The dataset is small and pediatric-only; the model and development risk thresholds require independent validation. All outputs need qualified clinical review and must not be used as standalone medical advice.
