import os
import pandas as pd
import torch
import re
import numpy as np
import warnings
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration - UPDATED PARAMETERS
CONFIG = {
    "model_name": "Saibo-creator/legal-roberta-base",
    "labels": {"negotiable": 0, "court": 1},
    "confidence_threshold": 0.5,  # Lowered from 0.7
    "max_length": 512,
    "temperature": 0.8  # Added for confidence calibration
}

# IMPROVED Legal Pattern Matchers
COURT_PATTERNS = re.compile(
    r"\b(court ordered|judge ruled|motion granted|injunction issued|summarily judgment|compelled arbitration)\b", 
    re.I
)
NEGOTIABLE_PATTERNS = re.compile(
    r"\b(mutually agreed|mediated settlement|revised terms|bilateral agreement|consensus reached|adjusted thresholds)\b", 
    re.I
)
STATUTORY_REFS = re.compile(
    r"\b(UCC §\d+|CFR \d+\.\d+|article \d+\.\d+|rule \d+[A-Z]*)\b", 
    re.I
)

@dataclass
class CaseExample:
    """Data class for structured case examples"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor = None
    id: str = None
    text: str = None

class LegalDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.has_labels = "label" in self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        case_id = str(self.data.iloc[idx]["id"])
        
        # Label handling
        label = None
        if self.has_labels:
            raw_label = self.data.iloc[idx]["label"]
            if pd.isna(raw_label):
                raise ValueError(f"Missing label in row {idx}")
                
            clean_label = str(raw_label).strip().lower()
            if clean_label not in CONFIG["labels"]:
                raise ValueError(f"Invalid label '{raw_label}' in row {idx}")
                
            label = torch.tensor(CONFIG["labels"][clean_label])

        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        )
        
        return CaseExample(
            input_ids=encoding["input_ids"].squeeze(0),
            attention_mask=encoding["attention_mask"].squeeze(0),
            labels=label,
            id=case_id,
            text=text
        )

def collate_fn(batch):
    """Custom collation function for data loader"""
    return {
        "input_ids": torch.stack([ex.input_ids for ex in batch]),
        "attention_mask": torch.stack([ex.attention_mask for ex in batch]),
        "labels": torch.stack([ex.labels for ex in batch]) if batch[0].labels is not None else None
    }

def decision_tree(text):
    """More precise rule-based fallback"""
    if STATUTORY_REFS.search(text):
        if COURT_PATTERNS.search(text):
            return "court"
        if NEGOTIABLE_PATTERNS.search(text):
            return "negotiable"
 
    if "agreement" in text.lower():
        if "breach" in text.lower() or "court" in text.lower() or "judge" in text.lower():
            return "court"
        else:
            return "negotiable"
    
    return "negotiable" 

def predict(model, tokenizer, text):
    """Hybrid prediction with temperature scaling"""
    device = model.device 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"]).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
   
    scaled_logits = outputs.logits / CONFIG["temperature"]
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    confidence, pred = torch.max(probs, dim=1)
    
    if confidence.item() < CONFIG["confidence_threshold"]:
        return decision_tree(text), confidence.item(), "rules"
    return ("court" if pred.item() == 1 else "negotiable"), confidence.item(), "model"

def main(csv_path):
  
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        valid_labels = {"court", "negotiable"}
        df_labels = df["label"].str.strip().str.lower()
        invalid = df[~df_labels.isin(valid_labels) | df["label"].isna()]
        if not invalid.empty:
            print("Invalid labels found:")
            print(invalid)
            raise SystemExit("Fix labels before training")

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")


    model_path = "./legal_model" if os.path.isdir("./legal_model") else CONFIG["model_name"]
    
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
 
    model.config.problem_type = "single_label_classification"
    model.config.id2label = {v: k for k, v in CONFIG["labels"].items()}
    model.config.label2id = CONFIG["labels"]

    dataset = LegalDataset(csv_path, tokenizer)
    

    if dataset.has_labels:
        print("\nTraining mode activated")
        args = TrainingArguments(
            output_dir="legal_model",
            per_device_train_batch_size=2,  
            num_train_epochs=5,            
            learning_rate=3e-5,            
            weight_decay=0.01,
            warmup_ratio=0.1,
            evaluation_strategy="no",
            save_strategy="epoch",
            logging_dir="./logs",
            remove_unused_columns=False,
            gradient_accumulation_steps=2,  
            fp16=torch.cuda.is_available()  
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collate_fn
        )
        trainer.train()
        
        model.save_pretrained("legal_model")
        tokenizer.save_pretrained("legal_model")
        print("\nModel saved with files:", os.listdir("legal_model"))

    model.eval()
    results = []
    for case in dataset:
        prediction = predict(model, tokenizer, case.text)
        results.append({
            "id": case.id,
            "text": case.text[:200] + "..." if len(case.text) > 200 else case.text,
            "category": prediction[0],
            "confidence": f"{prediction[1]:.2%}",
            "source": prediction[2]
        })

   
    confidences = [float(r['confidence'].strip('%'))/100 for r in results]
    print(f"\nConfidence Statistics:")
    print(f"Average: {np.mean(confidences):.2%}")
    print(f"Median: {np.median(confidences):.2%}")
    print(f"Minimum: {np.min(confidences):.2%}")

    pd.DataFrame(results).to_csv("categorized_cases2.csv", index=False)
    print(f"\nCategorized {len(results)} cases. Results saved to categorized_cases2.csv")

    test_texts = [
        "The court granted summary judgment under FRCP 56 after failed mediation.",
        "The parties reached an agreement to settle the dispute.",
        "The agreement was breached, and the court issued an injunction.",
        "The parties mutually agreed to revise the terms of the contract."
    ]
    for text in test_texts:
        pred = predict(model, tokenizer, text)
        print(f"Text: {text}\nPrediction: {pred}\n")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "caselaw_v2_small_train.csv")