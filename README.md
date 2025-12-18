# Analysis-of-Data-Sets-With-Learning-Conflicts-for-Machine-Learning

# 🧠 Learning Conflict Analysis Platform for Machine Learning

## 📌 Overview

This project implements and extends the research concept of **Learning Conflict Analysis** in supervised machine learning datasets.
Learning conflicts occur when **similar input samples map to significantly different output values**, causing confusion during model training and degrading performance.

The original research paper proposed a **distance-based conflict detection framework** to identify and remove such conflicting samples.
This project **reproduces the core methodology** and **extends it into a full-fledged, interactive platform** that can be used on **real-world datasets**.

---

## 📄 Research Background

### What the Original Paper Implemented

The paper introduced a systematic approach to:

1. Normalize dataset features
2. Compute **input difference (δᵢⱼ)** using Euclidean distance
3. Compute **target difference (Tᵢⱼ)**
4. Define **learning conflict scores**
5. Identify samples with high conflict
6. Remove highly conflicting samples
7. Evaluate performance improvement using RMSE

The paper demonstrated that **learning conflict removal improves regression model performance** more effectively than traditional outlier detection methods.

---

## 🚀 What This Project Adds (Improvements)

This project goes **far beyond static experimental reproduction**:

### 🔹 1. End-to-End Interactive Platform

* User uploads **any dataset**
* User selects **target column**
* Entire pipeline runs automatically

### 🔹 2. Robust Dataset Validation

* Detects supervised vs unsupervised datasets
* Ensures numeric target (regression-only)
* Validates feature availability
* Handles zero-variance targets
* Enforces minimum dataset size

### 🔹 3. Real-World Data Handling

* Automatic **missing value (NaN) detection**
* User-selectable imputation strategies
* Categorical feature detection
* Binary encoding (yes/no → 0/1)
* One-hot encoding for multi-category features
* Boolean feature support

### 🔹 4. Conflict vs Traditional Cleaning

* Compares **learning conflict removal**
* Against **IQR-based outlier removal**
* Demonstrates that conflict removal captures **subtle contradictions**, not just extreme values

### 🔹 5. Explainable & Downloadable Results

* RMSE before vs after conflict removal
* Conflict distribution visualization
* Downloadable:

  * Cleaned dataset
  * Conflict-scored dataset

### 🔹 6. Production-Ready Engineering

* Modular codebase
* Streamlit UI
* Session-state handling
* Reproducible environment
* Deployment-ready structure

---

## 🏗️ Project Structure

```text
learning_conflict_system/
│
├── learning_conflict_project/      # Original research & experiments
│
├── learning_conflict_platform/     # User-facing Streamlit application
│   ├── app.py
│   ├── core/
│   │   ├── validator.py
│   │   ├── preprocessing.py
│   │   ├── normalization.py
│   │   ├── cleaning.py
│   │   ├── modeling.py
│   │   └── visualization.py
│   └── requirements.txt
│
└── README.md
```

---

## 🧪 Supported Dataset Types

| Dataset Type                | Supported       |
| --------------------------- | --------------- |
| Numeric regression          | ✅ Yes           |
| Mixed categorical + numeric | ✅ Yes           |
| Boolean features            | ✅ Yes           |
| Missing values (NaN)        | ✅ Yes           |
| Classification datasets     | ❌ Not supported |
| Unsupervised datasets       | ❌ Not supported |

---

## 📊 Example Datasets (Kaggle)

* **Boston Housing Dataset** (numeric-only regression)
* **California Housing Dataset**
* **Housing Price Prediction Dataset**

---

## 🛠️ How to Run the Project Locally

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/pradyumna-11/Analysis-of-Data-Sets-With-Learning-Conflicts-for-Machine-Learning.git
cd learning-conflict-system/learning_conflict_platform
```

---

### ✅ 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

---

### ✅ 4. Run the Application

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧭 How to Use the Platform

1. Upload a CSV dataset
2. Select the target column
3. Handle missing values (if any)
4. Handle categorical features (if any)
5. Run learning conflict analysis
6. View RMSE improvement
7. Download cleaned datasets

---

## 📈 Evaluation Metric

* **Root Mean Squared Error (RMSE)**
  Used to evaluate regression performance **before and after conflict removal**.

---

## 🧠 Key Insight

> Learning conflict removal identifies *contradictory samples* that traditional outlier detection methods fail to capture, leading to more stable and accurate regression models.

---

## 🎓 Academic Value

* Research-based implementation
* Extended experimentation
* Real-world usability
* Suitable for:

  * Final-year project
  * Research continuation
  * ML system demonstrations

---

## 🧑‍💻 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

---

## 🔮 Future Enhancements

* PDF report generation
* Support for classification conflicts
* Advanced imputation strategies
* Deployment on Streamlit Cloud
* Automated experiment logging

---

## 📜 License

This project is intended for **academic and educational use**.

---

## 🙌 Acknowledgment

This project is inspired by and extends the research work on **Learning Conflict Analysis in Supervised Machine Learning Datasets**, transforming theoretical concepts into a practical, user-driven platform.
