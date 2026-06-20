# 🛡️ IntelliSpam: Email & SMS Spam Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2.0%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.22.0%2B-red.svg)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline designed to clean, preprocess, classify, and analyze SMS and email messages. This repository implements text cleaning (stemming, tokenization, stopword removal), feature extraction (TF-IDF), and compares multiple classifiers (**Multinomial Naive Bayes**, **Logistic Regression**, **Random Forest**, and **Support Vector Machines**) to achieve a high-precision classification system.

The selected model is deployed via an interactive, modern **Streamlit web dashboard** and a **CLI application**, and the project results are automatically compiled into a formatted **Microsoft Word report**.

---

## ✨ Key Features

- **Robust Preprocessing Pipeline:** Automatic text cleaning, including lowercasing, alphanumeric filtering, NLTK tokenization, English stopword removal, and Porter Stemming.
- **Model Comparison Suite:** Trains, evaluates, and compares four ML models on stratified training sets.
- **High-Precision Focus:** Optimizes classification metrics to keep False Positives low (ensuring legitimate emails are not lost in the spam folder).
- **Interactive Streamlit Web Dashboard:** Modern, glassmorphism-inspired UI featuring custom CSS, live predictions with confidence levels, and visualization tabs.
- **Jupyter Notebook EDA:** Dedicated notebook for Exploratory Data Analysis with class balance checks, message length distribution plots, and word clouds.
- **CLI Utility:** Predict spam/ham status on single sentences or run an interactive terminal loop.
- **Auto-generated Report:** A script to build a professional `report.docx` programmatically with all the metrics and evaluation tables.

---

## 📁 Directory Structure

```
Email-Spam-Detection/
│
├── dataset/
│   └── spam.csv                  # Raw dataset (SMS Spam Collection)
│
├── notebooks/
│   └── EDA.ipynb                 # Jupyter Notebook for Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py     # Text cleaning, tokenization, and stemming
│   ├── feature_extraction.py     # TF-IDF Vectorizer configuration
│   ├── model_training.py         # Trains models and selects the best classifier
│   ├── model_evaluation.py       # Evaluates the best model and generates plots
│   ├── predict.py                # Command-Line prediction utility
│   └── generate_report.py        # Generates report.docx programmatically
│
├── models/
│   └── spam_classifier.pkl       # Saved model pipeline (TF-IDF + SVM)
│
├── outputs/
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── accuracy_graph.png        # Bar chart comparing classifier performances
│   ├── distribution.png          # Distribution of ham vs spam in the dataset
│   └── wordcloud.png             # Word cloud of spam keywords
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
├── report.docx                   # Word document project report
└── README.md                     # This project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Email-Spam-Detection.git
cd Email-Spam-Detection
```

### 2. Create and Activate Virtual Environment
*On Windows (PowerShell):*
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*On macOS/Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Run the ML Pipeline

### Step 1: Preprocess Data & Train Models
Train the classifiers and save the best pipeline (Support Vector Machine) to the `models/` directory:
```bash
python src/model_training.py
```

### Step 2: Run Model Evaluation
Generate the performance charts (Confusion Matrix, Model Comparison Bar Graph) and save them to the `outputs/` folder:
```bash
python src/model_evaluation.py
```

### Step 3: Generate Word Report
Compile the project details and test results into a formal Microsoft Word report (`report.docx`):
```bash
python src/generate_report.py
```

---

## 🖥️ Usage & Applications

### 1. Launch the Streamlit Web Application
Run a premium, interactive web dashboard in your browser:
```bash
streamlit run app.py
```
*Dashboard features include:*
- **Live Classification Tab:** Enter text to get instant prediction (SPAM/HAM) with confidence levels.
- **Metrics Tab:** View performance comparison tables, confusion matrices, and model comparison graphs.
- **Insights Tab:** Explore the class distributions and keyword word clouds.

### 2. Run Command-Line Predictions
- **Single Prediction:**
  ```bash
  python src/predict.py --message "URGENT! Your mobile number has won a £2,000 bonus prize. Claim now by calling 09066364589."
  ```
- **Interactive CLI Loop:**
  ```bash
  python src/predict.py
  ```
  *(Type `exit` or `quit` to close the terminal session).*

---

## 📊 Model Evaluation Results

Models were evaluated on a **20% stratified test set** (964 Ham and 149 Spam messages) to ensure balanced representation.

| Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **97.93%** | **99.22%** | **85.23%** | **91.70%** |
| **Random Forest** | 97.93% | 100.00% | 84.56% | 91.64% |
| **Multinomial Naive Bayes (MNB)** | 97.39% | 99.18% | 81.21% | 89.30% |
| **Logistic Regression** | 96.95% | 98.32% | 78.52% | 87.31% |

### Key Observations:
- **Best Classifier:** The **Support Vector Machine (SVM)** achieved the highest F1-Score of **91.70%** and an overall accuracy of **97.93%**.
- **Precision vs. Recall:** While Random Forest achieved 100% precision, SVM achieved **99.22%** precision with a significantly higher recall (**85.23%** vs. **84.56%**), making it a much more balanced detector for real-world scenarios.

---

## 📈 Visualizations (Sample Outputs)

### Class Distribution (Imbalance)
<p align="center">
  <img src="outputs/distribution.png" width="450" alt="Class Distribution" />
</p>

### Confusion Matrix
<p align="center">
  <img src="outputs/confusion_matrix.png" width="400" alt="Confusion Matrix Heatmap" />
</p>

### Spam Word Cloud
<p align="center">
  <img src="outputs/wordcloud.png" width="550" alt="Spam Keywords Wordcloud" />
</p>

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

Developed by **[Your Name]** - feel free to reach out or contribute!
- **GitHub:** [@yourusername](https://github.com/your-username)
- **LinkedIn:** [Your Name](https://linkedin.com/in/your-profile)
