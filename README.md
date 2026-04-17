# 🚀 AI-Based Cloud Cost Optimization System

## 📌 Overview
This project is an AI-based system that detects inefficient cloud resource usage and generates cost optimization recommendations using Machine Learning and a Mini LLM.

---

## ⚙️ Features
- Machine Learning Model (Random Forest)
- Mini LLM (Markov-based text generation)
- AWS S3 Integration (Input + Output)
- Streamlit Dashboard
- Automated Report Generation

---

## 🧠 Technologies Used
- Python
- Scikit-learn
- AWS S3
- Streamlit
- Git & GitHub

---

## 📊 Dataset
Cloud resource dataset containing:
- CPU usage
- Memory usage
- Disk I/O
- Latency
- Throughput
- Cost

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
python train_model.py