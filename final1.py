import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F

label2id = {"negotiable": 0, "court": 1}
id2label = {0: "Negotiable", 1: "Court"}

def extract_judge(text):
    """
    Extracts the judge's name from the text by looking for the marker "CORAM:".
    Returns "Unknown" if the marker is not found.
    """
    marker = "CORAM:"
    if marker in text:
        part = text.split(marker)[1]
        judge_line = part.split("\n")[0]
        return judge_line.strip()
    else:
        return "Unknown"

class LegalCaseDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.has_labels = "label" in self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, "text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.has_labels:
            raw_label = self.data.loc[idx, "label"]
            if isinstance(raw_label, str):
                raw_label = raw_label.strip().lower()
                label = label2id.get(raw_label, 0)
            else:
                label = int(raw_label)
        else:
            label = 0  # Default if missing
        item["labels"] = torch.tensor(label, dtype=torch.long)
        if "source" in self.data.columns:
            item["case_name"] = self.data.loc[idx, "source"]
        else:
            item["case_name"] = self.data.loc[idx, "id"]
        item["id"] = self.data.loc[idx, "id"]
        item["original_text"] = text
        return item

def main():
    print("Starting training process...")

    tokenizer = RobertaTokenizer.from_pretrained("Saibo-creator/legal-roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("Saibo-creator/legal-roberta-base", num_labels=2)


    dataset = LegalCaseDataset("categorized_cases.csv", tokenizer, max_length=512)
    
    if dataset.has_labels:
     
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,
            evaluation_strategy="steps",
            eval_steps=10,
            save_steps=10,
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        print("Training complete.")
        trainer.evaluate()

        model.save_pretrained("./legal-case-classifier")
        tokenizer.save_pretrained("./legal-case-classifier")
    else:
        print("No labels found in CSV. Skipping training and evaluation.")

    results = []
    for i in range(len(dataset)):
        example = dataset[i]
        inputs = tokenizer(example["original_text"], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
 
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        priority_score = probs[predicted_class]
        predicted_category = id2label[predicted_class]
        
        case_name = example["case_name"]
        judge = extract_judge(example["original_text"])
        short_text = example["original_text"][:200] + "..." if len(example["original_text"]) > 200 else example["original_text"]
        
        results.append({
            "case_name": case_name,
            "judge": judge,
            "short_text": short_text,
            "predicted_category": predicted_category,
            "priority_score": priority_score
        })

    df_results = pd.DataFrame(results)

    court_cases = df_results[df_results["predicted_category"] == "Court"].sort_values(by="priority_score", ascending=False)
    negotiable_cases = df_results[df_results["predicted_category"] == "Negotiable"].sort_values(by="priority_score", ascending=False)
    final_df = pd.concat([court_cases, negotiable_cases], ignore_index=True)

    final_info_df = final_df[["case_name", "judge", "short_text", "priority_score", "predicted_category"]]
    final_info_df.to_csv("final_important_info_final.csv", index=False)

    print("Final important info saved to 'final_important_info_final.csv'.")

if __name__ == "__main__":
    main()
