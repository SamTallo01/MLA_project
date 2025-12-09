# MLA Project - WSI Classification with CLAM

This project implements a Whole Slide Image (WSI) classification pipeline using the **CLAM (Clustering-constrained Attention Multiple instance learning)** model. It includes tools for feature extraction, dataset preparation, training, and evaluation.

## 1. Setup

### Environment
Ensure you have a Python environment set up with the necessary dependencies.

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install torch torchvision torchaudio pandas matplotlib h5py scikit-learn xlrd
# Or use requirements.txt if available
pip install -r requirements.txt
```

### Directory Structure
- `dataset/`: Contains raw slides (`.ndpi`, `.svs`) and the diagnosis Excel file (`diagnosi.xls`).
- `feature_extractor/`: Contains code and output from the feature extraction step.
- `models/`: Contains the CLAM model implementation.
- `results/`: Directory where trained models and checkpoints will be saved.
- `dataset.py`: Data loader logic.
- `train.py`: Main training script.
- `eval.py`: Evaluation/Inference script.
- `create_dataset_csv.py`: Helper to generate the dataset CSV from the Excel file.

---

## 2. Pipeline Workflow

### Step 1: Feature Extraction
**Goal:** Convert WSI slides into feature vectors (patches).
- **Script:** `extract_fv.py`
- **Output:** `.pt` files (PyTorch tensors) in your features directory.
- *Status: Already completed.*

### Step 2: Dataset Preparation
**Goal:** Map slides to labels.
- **Script:** `create_dataset_csv.py`
- **Input:** `dataset/diagnosi.xls`
- **Output:** `dataset.csv` (contains `slide_id` and `label`) and `label_mapping.txt`.
- **Usage:**
  ```bash
  python create_dataset_csv.py
  ```
- *Status: Already completed. `dataset.csv` is ready.*

### Step 3: Training
**Goal:** Train the CLAM model.
- **Script:** `train.py`
- **Usage:**
  ```bash
  python train.py --csv_path dataset.csv --data_root_dir path/to/features --results_dir results --max_epochs 200 --model_type clam_sb
  ```
- **Key Arguments:**
  - `--csv_path`: Path to `dataset.csv`.
  - `--data_root_dir`: Path to the folder containing `.pt` feature files.
  - `--model_type`: `clam_sb` (Small) or `clam_mb` (Multi-branch).

### Step 4: Evaluation
**Goal:** Test a trained model.
- **Script:** `eval.py`
- **Usage:**
  ```bash
  python eval.py --csv_path dataset.csv --data_root_dir path/to/features --checkpoint_path results/checkpoint_fold_0.pt
  ```
- **Output:** Prints Accuracy, AUC, and Confusion Matrix.

---

## 3. Label Mapping
Refer to `label_mapping.txt` to understand the class labels (0, 1, etc.) generated from your diagnosis file.
