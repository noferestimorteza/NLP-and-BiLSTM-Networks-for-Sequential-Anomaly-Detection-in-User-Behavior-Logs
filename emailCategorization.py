import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
import csv

BATCH_SIZE = 32  
INPUT_FILE = 'UEBA/output_email_ACM2278.csv'
OUTPUT_FILE = 'UEBA/Categorized_email_ACM2278.csv'
specific_user = 'ACM2278'

device = 0 if torch.cuda.is_available() else -1
torch.backends.cudnn.benchmark = True  
print(f"Using {'GPU' if device == 0 else 'CPU'} for processing")

CONTENT_CATEGORIES = [
    'Job Apply',   
    'Personal',    
    'Social Media', 
    'Internal Comunication',   
    'News/Info',  
    'Miscellaneous'
]

class ContentDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

content_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32,  
    batch_size=BATCH_SIZE
)

def batch_categorize(texts):
    """Process a batch of texts at once"""
    if not texts:
        return []
    
    valid_texts = [str(text).strip() for text in texts if not pd.isna(text) and str(text).strip()]
    if not valid_texts:
        return ['unknown'] * len(texts)
    
    results = content_classifier(valid_texts, CONTENT_CATEGORIES, multi_label=False)
    
    output = []
    result_idx = 0
    for text in texts:
        if pd.isna(text) or not str(text).strip():
            output.append('unknown')
        else:
            output.append(results[result_idx]['labels'][0])
            result_idx += 1
    return output

print("Processing data...")
processed_data = []

with open(INPUT_FILE, 'r') as f:
    reader = csv.DictReader(f)
    relevant_rows = [row for row in reader if row['user'] == specific_user]

for i in tqdm(range(0, len(relevant_rows), BATCH_SIZE), desc="Processing batches"):
    batch = relevant_rows[i:i+BATCH_SIZE]
    texts = [row['content'] for row in batch]
    categories = batch_categorize(texts)
    
    for row, category in zip(batch, categories):
        row['content_category'] = category
        processed_data.append(row)

final_df = pd.DataFrame(processed_data)
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Processing complete! Saved to {OUTPUT_FILE}")