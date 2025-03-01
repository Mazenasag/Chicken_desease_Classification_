# 🐔 Chicken Disease Classification
![Alt Text](static/PrediChick.PNG)

## 📌 Problem Statement

This project focuses on classifying chicken health conditions into **Healthy** and **Coccidiosis** using deep learning. The objective is to fine-tune a **VGG16** model to achieve high classification accuracy, assisting poultry farmers in early disease detection.

## 📊 Data Collection

**Dataset Source:** [Specify dataset source, e.g., Kaggle or a custom dataset]
https://www.kaggle.com/datasets/allandclive/chicken-disease-1

### Features Description:

| Feature   | Description                                            |
| --------- | ------------------------------------------------------ |
| **Image** | Image of the chicken (Healthy or Coccidiosis affected) |
| **Label** | Healthy (Normal) or Coccidiosis (Infected)             |

### Dataset Summary:

- Binary classification problem (**Healthy** vs. **Coccidiosis**).
- Preprocessed images with **resizing** and **normalization**.
- **Data Augmentation** applied to improve model generalization.

## 📈 Model Performance

| Model                  | Accuracy  | Precision | Recall    | F1-Score  |
| ---------------------- | --------- | --------- | --------- | --------- |
| **VGG16 (Fine-tuned)** | **93.8%** | **92.5%** | **94.2%** | **93.3%** |

## 🔍 Insights & Key Findings

- **VGG16 fine-tuning** yielded **93.8% accuracy** for classifying chicken health.
- **Data augmentation** improved model robustness.
- **Transfer learning** helped achieve high performance with limited data.

## 📊 Exploratory Data Analysis (EDA)

🔗 [EDA Notebook](#)

## 🏗️ Model Training Approach

🔗 [Model Training Notebook](.research\04_model_evaluation.ipynb)

## 🔍 Model Interpretation with Grad-CAM

🔗 [Grad-CAM Visualization](#)

## 🖥️ Screenshot of UI

(Include an image here)

## 🛠️ How to Run the Project

### Clone the repository:

```bash
git clone https://github.com/Mazenasag/Chicken_Disease_Classification.git
cd Chicken_Disease_Classification


## Workflows

1. Update the config Yaml file
2. Update the config secrets file [optional]
3. Update the params.yaml
4. Update the entity
5. Update the params.yaml
6. Update the configuration manager in src config
7. Update the compenents
8. Update the main.py
9. Update the dvc.yaml
```
