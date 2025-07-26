import pandas as pd
from datasets import load_dataset

dataset = load_dataset("DanKamNdi/caselaw-v2-small")

df = pd.DataFrame(dataset["train"])

df.to_csv("caselaw_v2_small_train.csv", index=False)

print("CSV file saved as 'caselaw_v2_small_train.csv'")
