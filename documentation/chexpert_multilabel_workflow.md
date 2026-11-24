# Train the 14-finding CheXpert model

The project supports these 14 CheXpert observations: Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, and No Finding. Findings are predicted independently, so an image can have several positive probabilities.

## Dataset to obtain

Use the **CheXpert v1.0 chest radiograph dataset** (the small release is sufficient to start) from Stanford or a Kaggle mirror that explicitly contains CheXpert v1.0 images and its matching 14-observation annotation CSV. Check the dataset page, terms, and citation before downloading. Do not substitute the separate pediatric Kaggle Pneumonia dataset: it does not have the 14 CheXpert labels.

The local workspace may contain `dataset/chexpert/train_cheXbert.csv`, with CheXpert-style paths and label columns, but dataset manifests and images are excluded from the public Git repository. Obtain the matching X-ray images and annotation CSV from a permitted source and extract them so paths such as `CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg` exist relative to the project root. If you use another mirror, its image paths and annotation CSV must correspond; use its annotation CSV as `--source` when creating splits.

The public Git repository does not include X-ray image files or trained checkpoints. You must download the licensed images before running training. Storage and training time depend on the release, hardware, and batch size; a GPU is recommended.

## Create patient-separated manifests

From the project root in PowerShell:

```powershell
$PYTHON = Join-Path $PWD ".venv\Scripts\python.exe"
& $PYTHON dataset\chexpert\create_split.py --source dataset\chexpert\train_cheXbert.csv
```

This makes 70/15/15 train, validation, and test CSVs separated by patient. If a Kaggle mirror uses a different CSV, pass its path to `--source`. The CSV must contain a `Path` column and the 14 CheXpert label columns; paths must contain patient IDs.

## Train and evaluate

```powershell
$PYTHON = Join-Path $PWD ".venv\Scripts\python.exe"
& $PYTHON "ai model\train_chexpert.py" --train dataset\chexpert\train.csv --val dataset\chexpert\val.csv --image-root . --epochs 5 --batch-size 16 --pretrained --uncertain-label-policy zeros --output "ai model\checkpoints\chexpert_densenet121.pt"
```

`--pretrained` initializes DenseNet-121 from ImageNet weights and downloads them the first time if they are not cached; omit it to train from random initialization. CheXpert uses `-1` for uncertain labels and blank values for unreported observations. The default `zeros` policy treats uncertain labels as negative; blank labels are excluded from the loss. `--uncertain-label-policy ones` or `ignore` are available alternatives. Record the choice when reporting results.

Evaluate the held-out patients before using the model:

```powershell
$PYTHON = Join-Path $PWD ".venv\Scripts\python.exe"
& $PYTHON "ai model\evaluate.py" --test dataset\chexpert\test.csv --image-root . --checkpoint "ai model\checkpoints\chexpert_densenet121.pt"
```

The backend automatically uses `ai model/checkpoints/chexpert_densenet121.pt` when that checkpoint exists and its labels match. Otherwise, it retains the existing binary pneumonia model/demo. Restart the backend after adding or replacing a checkpoint. The page status will show “14-finding model ready” when the multi-label model is selected.

## Limits

Training a model does not make it perfect or clinically valid. Report per-finding test metrics, evaluate calibration and performance on an independent dataset, and obtain clinical review. The prototype severity numbers and triage thresholds are illustrative only; the app is research software, not a diagnostic tool.

## Run the web app on Google Cloud Run

The project root includes a Dockerfile that serves both the frontend and FastAPI backend from one Cloud Run service. It packages the CheXpert checkpoint and excludes the local datasets. The deployed inference runs on CPU; the optional report generator uses its structured fallback unless an Ollama service is separately configured.

After installing the Google Cloud CLI, signing in, selecting the project, and ensuring it has an **open, active billing account**, run these commands from the project root:

```powershell
gcloud config set project YOUR_PROJECT_ID
gcloud run deploy ai-imaging-triage `
  --source . `
  --region asia-south1 `
  --memory 2Gi `
  --cpu 2 `
  --concurrency 1 `
  --max-instances 1 `
  --allow-unauthenticated
```

Cloud Build uses the Dockerfile to build the service. When deployment finishes, use the service URL printed by `gcloud run deploy`. The `--allow-unauthenticated` option makes the app publicly reachable; only use it for non-clinical demos with non-identifying images. For a restricted service, omit that option and grant Cloud Run Invoker access only to intended users. The instance and request limits reduce resource use but do not guarantee a zero bill.
