# TransformerGP
TransformerGP is a PyTorch-based deep learning framework specifically designed for plant genomics, with a focus on rice genetic analysis. By integrating multi-modal data such as Single Nucleotide Polymorphisms (SNPs) and Gene Expression (EXP), this project utilizes **1D-CNN feature extraction**, **Transformer encoders**, and a **Gating Mechanism** to achieve high-accuracy predictions of complex agronomic traits.

Beyond robust predictive performance, this project integrates the **SHAP (SHapley Additive exPlanations)** algorithm to interpret model feature importance. It also provides automated scripts to construct **Gene Regulatory Networks (GRNs)**, which can be easily exported to Gephi for visualization and clustering analysis. This offers deep computational insights for Genome-Wide Association Studies (GWAS), genomic selection, and breeding strategies such as three-line hybrid systems.

## ✨ Key Features

* **Multi-modal Fusion Architecture**: Supports single-modality (EXP or SNP only) and dual-modality joint prediction, utilizing a gating network to adaptively learn the weights of different modalities.
* **End-to-End Data Processing**: Built-in pipeline for missing value imputation, outlier handling, feature pre-selection via `SelectKBest` (F-regression), and feature standardization.
* **Interpretability & Network Construction**: Not only outputs predictions but also calculates SHAP values for each genomic feature (e.g., specific LOC markers). It builds `gene-gene` and `snp-gene` network nodes and edges based on Pearson correlation and SHAP importance.
* **Comprehensive Baseline Comparisons**: Includes a fair cross-validation framework to compare the deep learning model against traditional machine learning algorithms (SVR, Lasso, RandomForest, CatBoost, etc.).

---

## 📂 Project Structure & File Descriptions

The project consists of the following Python scripts, each designed with high cohesion and low coupling:

### Core Models & Training
* **`main.py`**: The execution entry point of the project. It coordinates data loading, model initialization, the K-fold cross-validation loop, and calls the training and evaluation functions.
* **`model.py`**: Defines all PyTorch neural network architectures.
    * `ExpTraitPredictionModel` / `SnpTraitPredictionModel`: Single-modality networks based on CNN + Transformer.
    * `CombinedTraitPredictionModel`: A dual-branch network that uses a `torch.sigmoid` gating mechanism to fuse features before feeding them into the Transformer layers for global feature interaction.
* **`train.py`**: Contains the core logic for model training (forward pass, loss calculation, backward pass). It supports robust training with learning rate schedulers (`ReduceLROnPlateau`) and Early Stopping. It also includes the logic for extracting and saving SHAP values.

### Data & Configuration
* **`config.py`**: The global configuration file. It contains all file path definitions, model hyperparameters (e.g., hidden dimensions, number of attention heads), training parameters (learning rate, batch size, random seeds), and the list of target phenotypes (e.g., `Heading_date`, `Plant_height`, `Yield`, `Grain_length`, `Grain_weight`).
* **`data_loader.py`**: The data loading and preprocessing pipeline. It reads CSV data, aligns sample IDs, handles NaN/Inf outliers, applies `StandardScaler` and `SelectKBest`, and packages the data into PyTorch `DataLoader` or `TensorDataset` objects.

### Interpretability & Analysis Tools
* **`GRN.py`**: The Gene Regulatory Network builder. It reads the SHAP value summary files exported by `train.py` and the original feature matrices to calculate feature correlations, ultimately generating Gephi-compatible `nodes.csv` and `edges.csv` files.
* **`sklearn_models_eval.py`** & **`standard_compare.py`**: Scripts for baseline comparisons and ablation studies. They evaluate `scikit-learn` and various tree-based ensemble models on the same datasets to validate the performance advantages of the Transformer architecture.

### Utilities
* **`generate_toy_data.py`** (Optional): A script to generate a lightweight, synthetically generated "toy dataset" that perfectly matches the required input formats, making it easy to quickly verify the environment.

---

## 📊 Input Data Formatting

To bypass GitHub's file size limits and facilitate quick out-of-the-box testing, a lightweight "toy dataset" is provided in the `data/rice4k_219/` directory. It reflects the exact tabular structure of the real Rice4k dataset.

When applying your own full sequencing data, please ensure your files follow these formatting specifications:

### 1. Phenotype Data (`rice4k_ph.csv`)
Records continuous agronomic traits of the samples.
* **Required Column**: `ID` (Sample identifier, e.g., `RICE_0001`).
* **Other Columns**: Phenotype names corresponding strictly to the `PHENOTYPES` list in `config.py` (e.g., `Plant_height`, `Heading_date`). Missing values can be left blank (NaN).

### 2. Gene Expression Data (`rice4k_exp.csv`)
Transcriptome data matrix.
* **Required Column**: `ID`.
* **Other Columns**: Gene expression features (e.g., columns named `LOC_Os...`). Values should be normalized expression levels (like FPKM/TPM). The program will automatically handle minor NaN occurrences.

### 3. Single Nucleotide Polymorphism Data (`rice18k_sw.csv`)
Genomic variation data.
* **Required Column**: `ID`. (If exported from PLINK, columns like `FID`, `PAT`, `MAT`, `SEX` are compatible and will be automatically dropped by the loader).
* **Feature Columns**: SNP loci information, typically encoded as 0, 1, and 2 (representing different genotypes).

### 4. Cross-Validation Splits (`rice4k_fold8.json`)
Defines the K-fold splits to ensure that baseline comparisons and deep learning models evaluate the exact same data partitions.
* **Format**: A JSON array where each element contains `"fold"` (fold number), `"train_samples"` (list of training IDs), and `"test_samples"` (list of testing IDs).

---

## 🚀 Quick Start

### 1. Environment Setup
It is recommended to use Python 3.8+ with a CUDA-enabled GPU environment.
```bash
pip install torch numpy pandas scikit-learn shap
```

### 2. Run Model Training and Evaluation
After configuring your paths and hyperparameters in `config.py`, simply run `main.py`:
```bash
python main.py
```
This script will iterate through all phenotypes defined in `config.py`, perform cross-validation, and save the following to the `data/results/` directory:
* Summary of test set metrics (Pearson correlation, p-value, MSE, R²).
* Best model weights (`.pt` files) for each phenotype.
* SHAP feature importance (`.csv` files).

### 3. Build and Visualize Gene Regulatory Networks (GRN)
Once `main.py` has successfully generated the SHAP value files, run:
```bash
python GRN.py --phenotype Heading_date --model_type combined --top_exp 200 --include_snp
```
This will generate node and edge files in `data/results/grn/`. You can import these directly into **Gephi**, run Modularity or Leiden clustering algorithms, and visualize the complex gene-gene and snp-gene interaction networks.
