

# Legal Case Classification System

A **transformer-based legal NLP system** that classifies legal case texts into **Negotiable** or **Court-bound** categories using a fine-tuned **RoBERTa model** combined with rule-based fallback logic.

The system supports **batch CSV ingestion**, **confidence scoring**, and **prioritization of cases**, making it suitable for legal analytics and case triaging workflows.

---

# Overview

Legal case documents often contain large volumes of unstructured text.
This project builds an **automated classification pipeline** to determine whether a case is:

* **Negotiable** → can potentially be settled or mediated
* **Court** → requires judicial intervention

The system integrates:

* Transformer-based NLP classification
* Rule-based fallback decision logic
* Batch processing of CSV datasets
* Confidence scoring and prioritization
* A Flask web interface for uploading and classifying cases

---

# Key Features

* Transformer-based classification using **RoBERTa**
* Rule-based fallback for low-confidence predictions
* CSV batch processing pipeline
* Case prioritization using model confidence
* Judge name extraction from legal text
* Web interface built using **Flask**
* Supports large text inputs (up to 512 tokens)

---

# Project Architecture

```
project-root
│
├── app.py                    
├── final.py                  
│
├── templates/                
│   ├── index.html
│   └── results.html
│
├── static/                  
│
├── logs/                    
│
├── datasets
│   ├── categorized_cases.csv
│   ├── train_dataset.csv
│   └── eval_dataset.csv
│
└── README.md
```

---

# Model Architecture

The classifier uses **RoBERTa for sequence classification**.

Pipeline:

```
Legal Case Text
        ↓
Tokenizer (RoBERTa)
        ↓
Transformer Encoder
        ↓
Classification Layer
        ↓
Softmax Confidence
        ↓
Prediction
```

If the model confidence falls below a threshold, a **rule-based decision tree** is used.

---

# Rule-Based Fallback System

When model confidence is low, regex rules classify the case using legal patterns such as:

Court Indicators

```
court ordered
judge ruled
motion granted
injunction issued
```

Negotiable Indicators

```
mutually agreed
mediated settlement
bilateral agreement
```

Statutory References

```
UCC §...
CFR ...
Article ...
```

---

# Training Pipeline

The training pipeline is implemented in **`final.py`**.

Steps:

1. Load dataset from CSV
2. Tokenize case text
3. Split dataset (80% train / 20% evaluation)
4. Fine-tune **legal-roberta-base**
5. Evaluate performance
6. Generate ranked case predictions

The model outputs a **priority score** representing classification confidence.

---

# Example Output

After inference, cases are saved as:

```
final_important_info_final.csv
```

Example output structure:

| Case Name | Judge         | Category   | Priority Score |
| --------- | ------------- | ---------- | -------------- |
| Case A    | Justice Smith | Court      | 0.92           |
| Case B    | Justice Rao   | Negotiable | 0.81           |

Court cases are prioritized first based on confidence.

---

# Web Application

The Flask app allows users to upload CSV files containing legal case text.

Expected CSV format:

```
text
"The defendant breached the contract..."
"The parties mutually agreed to settle..."
```

API route:

```
POST /classify
```

Response:

```
[
 {
  "text": "...",
  "category": "court",
  "confidence": "92%",
  "source": "model"
 }
]
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/legal-case-classifier
cd legal-case-classifier
```

Install dependencies:

```
pip install flask torch transformers pandas
```

---

# Running the Web App

```
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

Upload a CSV file and view classification results.

---

# Training the Model

```
python final.py
```

# Future Improvements
* Add multi-class legal classification
* Improve dataset size and diversity
* Deploy as a cloud API
* Add legal summarization using LLMs
* Build interactive dashboard for case analytics

---

