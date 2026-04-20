# 🚀 AI-Based Cloud Cost Optimization System (AWS + AI)

## 📌 Overview
This project is an AI-based cloud cost optimization system that integrates Machine Learning, a Mini Language Model (LLM), and AWS cloud services.

The system analyzes cloud usage data, detects inefficient resources, and generates optimization suggestions using a serverless architecture.

---

## ⚙️ Features
- Machine Learning Model (Random Forest – local)
- Mini LLM (Markov-based suggestion generator)
- AWS S3 Integration (Input + Output storage)
- AWS Lambda (Serverless processing)
- Streamlit Dashboard (User interface)
- Automated Report Generation

---

## 🧠 Technologies Used
- Python  
- Scikit-learn  
- Streamlit  
- AWS S3  
- AWS Lambda  
- Boto3  
- Git & GitHub  

---

## 📊 Dataset
The dataset contains cloud usage metrics such as:
- CPU Usage  
- Memory Usage  
- Disk I/O  
- Latency  
- Throughput  
- Cost  

A target variable is used to classify resources as efficient or inefficient.

---

## 🏗️ Architecture
User (Streamlit)
↓
Upload JSON
↓
AWS S3 (Input Storage)
↓
AWS Lambda
↓
Mini LLM (Analysis)
↓
AWS S3 (Store Report)
↓
Streamlit Dashboard (Display)


---

## 🔁 Workflow

1. User uploads cloud usage data via Streamlit  
2. Data is uploaded to AWS S3  
3. Streamlit triggers AWS Lambda  
4. Lambda:
   - Fetches data from S3  
   - Runs LLM-based analysis  
   - Generates optimization suggestions  
5. Report is uploaded back to S3  
6. Streamlit fetches and displays results  

---

## ☁️ AWS Integration

### 📦 Amazon S3
- Stores input file: `cost_data.json`
- Stores output file: `cloud_cost_report.txt`
- Acts as centralized storage

---

### ⚡ AWS Lambda
- Serverless compute service  
- Processes data and generates reports  
- Integrates with S3 for input/output  

---

## 🤖 AI Components

### 🔹 Machine Learning (Local)
- Algorithm: Random Forest  
- Predicts inefficient resource usage  
- Saved as `cloud_model.pkl`  
- Not deployed in Lambda due to dependency limitations  

---

### 🔹 Mini LLM
- Markov-chain-based model  
- Generates human-readable optimization suggestions  
- Runs inside Lambda  

---

## ▶️ How to Run

### 1️⃣ Train ML Model
```bash
python model_ml.py
```
### 1️⃣ Run Dashboard
```bash
streamlit run dashboard.py
```
## 🐳 Run with Docker
```bash
docker run -p 8501:8501 yourusername/cloud-cost-app
```

---