## KPCA-Based Intrusion Detection System (IDS)

---

## 📌 Overview

This project presents a machine learning-based Intrusion Detection System (IDS) that uses Kernel Principal Component Analysis (KPCA) for nonlinear dimensionality reduction combined with multiple classification algorithms. The system focuses not only on detection accuracy but also on cost-sensitive decision making and prediction stability.

The proposed framework is evaluated on benchmark datasets (CICIDS and UNSW) and introduces a Decision Stability Index (DSI) to measure consistency across different data clusters.

---

## 🎯 Objectives
	•	Improve intrusion detection performance using KPCA
	•	Capture nonlinear patterns in network traffic data
	•	Reduce feature dimensionality while preserving important information
	•	Minimize misclassification cost using cost-sensitive threshold tuning
	•	Evaluate model stability using Decision Stability Index (DSI)
	•	Analyze performance across multiple train-test splits (0.2, 0.4, 0.6)

---

## 🧠 Methodology

1. Data Preprocessing
	•	Handling missing values
	•	Removing low-variance features
	•	Feature scaling using StandardScaler


2. Dimensionality Reduction
	•	Kernel Principal Component Analysis (KPCA)
	•	Supports multiple kernels:
	•	Linear
	•	RBF
	•	Polynomial
	•	Cosine
	•	Sigmoid
	•	Captures nonlinear relationships in data
	•	Reduces computational complexity


3. Machine Learning Models
	•	Decision Tree
	•	Random Forest
	•	Extra Trees
	•	Gradient Boosting
	•	AdaBoost
	•	K-Nearest Neighbors
	•	XGBoost


4. Cost-Sensitive Thresholding
	•	Assigns higher penalty to False Negatives
	•	Optimizes threshold based on:
	•	MCC
	•	Cost function
	•	Improves real-world detection reliability


5. Clustering-Based Analysis
	•	KMeans clustering
	•	Evaluates predictions across clusters
	•	Enables stability analysis


6. Evaluation Metrics
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score
	•	ROC-AUC
	•	Matthews Correlation Coefficient (MCC)
	•	Decision Stability Index (DSI)

---

## 📊 Datasets

Due to their large size, the datasets are not included in this repository. Download them from:

---

🔹 CICIDS Dataset
	•	Source: Canadian Institute for Cybersecurity
	•	Link: https://www.unb.ca/cic/datasets/ids-2017.html

---

🔹 UNSW-NB15 Dataset
	•	Source: University of New South Wales
	•	Link: https://research.unsw.edu.au/projects/unsw-nb15-dataset

⸻

## 📁 Folder Structure

datasets/
├── CICIDS_MASTER.csv
├── UNSW_MASTER.csv

Ensure file paths match your local setup.

---

📈 Key Results
	•	KPCA effectively captured nonlinear patterns in data
	•	XGBoost and Random Forest achieved top performance
	•	High detection accuracy (~99% on CICIDS)
	•	Strong MCC and DSI values, indicating robustness
	•	Cost-sensitive tuning reduced false negatives
	•	KPCA kernels (especially RBF/Poly) improved performance

---

## 🧪 Experimental Setup
	•	Train-test splits: 0.2, 0.4, 0.6
	•	KPCA components: 30
	•	Multiple kernels evaluated
	•	Cost-sensitive threshold optimization applied
	•	Cluster-based DSI computation

---

## 🔍 Key Findings
	•	KPCA improves detection by capturing nonlinear structures
	•	Accuracy alone is insufficient for IDS evaluation
	•	MCC and cost provide better performance insight
	•	DSI highlights prediction consistency across clusters
	•	Model performance varies with dataset complexity and split ratio

---

## 📌 Contribution
	•	KPCA-based IDS framework for nonlinear feature extraction
	•	Cost-sensitive decision mechanism for realistic deployment
	•	Decision Stability Index (DSI) for robustness evaluation
	•	Multi-kernel and multi-split experimental analysis

---

## 📄 Paper

This repository is associated with a research paper on KPCA-based intrusion detection with cost-sensitive learning and stability analysis.

---

## ⚙️ Requirements
	•	Python 3.x
	•	Scikit-learn
	•	NumPy
	•	Pandas
	•	XGBoost

---

## 🚀 Future Work
	•	Real-time intrusion detection system
	•	Deep learning integration (CNN, LSTM)
	•	Adaptive threshold optimization
	•	Deployment in production environments

---

## 👨‍💻 Author

Piyush Prateek
MTech Computer Science Engineering

---

## 📜 License

This project is open-source and available under the MIT License

---
