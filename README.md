# 🌿 TMRMed


*This paper is still **under review**, and more information (e.g., raw datasets) will be released after publication.*


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)


## 📁 1. Folder Specification

- `raw_data/`: clinical datasets
  - `data.csv`: Standard clinical data (symptoms, syndromes, prescriptions)
  - `TCM_Lung.xlsx`: Lung cancer dataset

- `processed_data/`: Preprocessed data files
  - `processed_data_discrete.pkl`: Discrete data (non-sequential)
  - `processed_data_continuous.pkl`: Sequential data (train/test)
  - `processed_tcm_lung_*.pkl`: Lung cancer processed data

- `model.py`: Core model implementations
  - `Ours`: Graph neural network with orthogonal constraints
  - `SimpleMLP`: Baseline MLP model

- `main.py`: Training and evaluation script
- `preprocess.py`: Data preprocessing for standard data
- `preprocess_tcm_lung.py`: Data preprocessing for lung cancer data
- `train.py`: Model training utilities
- `utils.py`: Evaluation metrics and visualization
- `results/`: Training results and visualizations
- `saved_model/`: Saved model checkpoints

## 🚀 2. Preliminary

### 2.1 📦 Package Dependency

Please install the environment according to the following version:

```bash
python == 3.8.17
torch == 2.0.1
numpy == 1.22.3
pandas == 2.0.2
scikit-learn == 1.3.0
matplotlib == 3.7.1
seaborn == 0.12.2
openpyxl == 3.1.2
```

### 2.2 📊 Data Processing

```bash
# Process our data
python preprocess.py

# Process lung cancer data
python preprocess_tcm_lung.py
```

### 2.3 🎯 Run the Code

```bash
# Quick start with our data
python main.py --model_name Ours --dataset Ours --embedding_dim 32 --num_epochs 30 --train True

# Run with lung cancer data
python main.py --model_name Ours --dataset tcm_lung --embedding_dim 64 --num_epochs 30 --train True
```

## 🤝 Contribution

We welcome contributions to improve TMRMed! Please feel free to submit issues or pull requests.
