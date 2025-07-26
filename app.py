from flask import Flask, json, request, render_template, jsonify, redirect, url_for
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import os
import pandas as pd

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  

CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "./legal_model"),
    "labels": {"negotiable": 0, "court": 1},
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
    "max_length": int(os.getenv("MAX_LENGTH", 512)),
    "temperature": float(os.getenv("TEMPERATURE", 0.8))
}

tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model_name"])
model = RobertaForSequenceClassification.from_pretrained(CONFIG["model_name"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

COURT_PATTERNS = re.compile(r"\b(court ordered|judge ruled|motion granted|injunction issued)\b", re.I)
NEGOTIABLE_PATTERNS = re.compile(r"\b(mutually agreed|mediated settlement|bilateral agreement)\b", re.I)
STATUTORY_REFS = re.compile(r"\b(UCC §\d+|CFR \d+\.\d+|article \d+\.\d+)\b", re.I)

def decision_tree(text):
    """Fallback classification using regex rules."""
    if STATUTORY_REFS.search(text):
        if COURT_PATTERNS.search(text):
            return "court"
        if NEGOTIABLE_PATTERNS.search(text):
            return "negotiable"

    if "agreement" in text.lower():
        if "breach" in text.lower() or "court" in text.lower():
            return "court"
        return "negotiable"

    return "negotiable"

def predict(text):
    """Model-based classification with fallback."""
    if not text.strip():
        return "invalid", 0.0, "error"  # Handle empty inputs
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"]).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits / CONFIG["temperature"], dim=-1)
        confidence, pred = torch.max(probs, dim=1)

        if confidence.item() < CONFIG["confidence_threshold"]:
            return decision_tree(text), confidence.item(), "rules"

        return ("court" if pred.item() == 1 else "negotiable"), confidence.item(), "model"
    except Exception as e:
        return "error", 0.0, str(e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "The CSV file is empty."}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400

    if "text" not in df.columns:
        return jsonify({"error": 'CSV must contain a "text" column.'}), 400
    
    results = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:  # Skip empty rows
            results.append({"text": text, "category": "invalid", "confidence": "0%", "source": "error"})
            continue

        category, confidence, source = predict(text)
        results.append({
            "text": text,
            "category": category,
            "confidence": f"{confidence:.2%}",
            "source": source
        })

    # Redirect to /results with the data
    return jsonify(results)  # This will be handled by the frontend

@app.route("/results", methods=["GET"])
def results():
    data = request.args.get("data")
    if data:
        data = json.loads(data)  # Parse the JSON data from the URL
        print("Data received from URL:", data)  # Debug statement
    else:
        data = []  

    return render_template("results.html", data=data)
if __name__ == "__main__":
    app.run(debug=True)
