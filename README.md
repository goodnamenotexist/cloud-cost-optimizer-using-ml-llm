# 🚀 AI-Based Cloud Cost Optimization System

## 📌 Overview
This project presents an AI-based system for optimizing cloud resource costs using Machine Learning and a Mini Language Model (LLM). The system analyzes cloud usage data to detect inefficient resources and generates intelligent optimization suggestions.

---

## ⚙️ Features
- 🔹 Machine Learning Model (Random Forest)
- 🔹 Mini LLM (Markov-based text generation)
- 🔹 AWS S3 Integration (Input + Output)
- 🔹 Streamlit Dashboard for visualization
- 🔹 Automated Report Generation

---

## 🧠 Technologies Used
- Python
- Scikit-learn
- AWS S3
- Streamlit
- Git & GitHub

---

## 📊 Dataset
The dataset includes cloud resource usage metrics such as:
- CPU Usage
- Memory Usage
- Disk I/O
- Latency
- Throughput
- Cost

A target variable is created based on utilization to classify resources as efficient or inefficient.

---

## 🔁 Workflow
1. Fetch cloud data from AWS S3  
2. ML model predicts inefficient resources  
3. Mini LLM generates optimization suggestions  
4. Report is generated and uploaded to S3  
5. Dashboard displays results  

---

## ▶️ How to Run

### 1. Train Model
```bash
python model_ml.py
```
### 2. Run Main System
```bash
python main.py
```
### 3. Run Dashboard
```bash
streamlit run dashboard.py
```

---
