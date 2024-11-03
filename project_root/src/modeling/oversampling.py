import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from sklearn.model_selection import train_test_split
from src.utils.file_utils import load_data, save_to_csv, load_config
from src.preprocessing.preprocess_text import replace_brand_words

# Define paths based on your local folder structure
RAW_DATA_PATH = "project_root/data/raw/"
PROCESSED_DATA_PATH = "project_root/data/processed/"
MODEL_SAVE_PATH = "project_root/models/"

# Downsample
def downsample_if_needed(df, max_rows=200000):
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=1) 
    return df

def prepare_data(data_frames, label):
    combined_df = downsample_if_needed(pd.concat(data_frames[label], ignore_index=True))
    combined_df['content'] = combined_df['content'].apply(replace_brand_words)
    reviews = combined_df['content'].tolist()
    train_reviews, val_reviews = train_test_split(reviews, test_size=0.1, random_state=42)
    return train_reviews, val_reviews

 # Fungsi untuk membuat DataLoader
def create_dataloader(tokenizer, train_reviews, val_reviews):
    train_dataset = ReviewDataset(train_reviews, tokenizer)
    val_dataset = ReviewDataset(val_reviews, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    return train_dataloader, val_dataloader

# GPT-2 Training Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=200):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        encoding = self.tokenizer(review, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

def train_model(model, model_name, train_dataloader, val_dataloader, optimizer, device, epochs=1, patience=2):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = inputs.clone().to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = inputs.clone().to(device)
                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_save_path = os.path.join(MODEL_SAVE_PATH, model_name)
            os.makedirs(model_save_path, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"Best model {model_name} saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def run_model_training(path_input_list):
    # Mengelompokkan data menjadi positive, neutral, dan negative
    data_frames = {"positive": [], "neutral": [], "negative": []}
    for input_path in path_input_list:
        df = load_data(input_path)
        data_frames["positive"].append(df[df.score > 3])
        data_frames["neutral"].append(df[df.score == 3])
        data_frames["negative"].append(df[df.score < 3])

    # Persiapan data untuk setiap label
    train_reviews = {}
    val_reviews = {}
    for label in ["positive", "neutral", "negative"]:
        train_reviews[label], val_reviews[label] = prepare_data(data_frames, label)

    # Inisialisasi tokenizer dan model GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('cahya/gpt2-small-indonesian-522M')
    tokenizer.pad_token = tokenizer.eos_token  # Set <eos> as padding token if not defined
    model_base = GPT2LMHeadModel.from_pretrained('cahya/gpt2-small-indonesian-522M')
    model_base.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_base.to(device)

    # Melatih model untuk setiap label dengan early stopping
    for label in ["positive", "neutral", "negative"]:
        print(f"\nStarting training for {label} reviews...")
        
        # Buat DataLoader untuk setiap label
        train_dataloader, val_dataloader = create_dataloader(tokenizer, train_reviews[label], val_reviews[label])

        # Salin model untuk setiap kelas sentimen
        model = model_base.__class__.from_pretrained('cahya/gpt2-small-indonesian-522M')
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=1e-5)

        # Jalankan pelatihan
        train_model(
            model, f"GPT2_best_model_{label}", 
            train_dataloader, 
            val_dataloader, 
            optimizer,
            device
        )
    
if __name__ == "__main__":
    # Load paths from config.yaml
    config = load_config()
    
    # Create input_path_list from YAML file paths
    input_path_list = list(config["oversampling"].values())
    
    # Run model training with the loaded paths
    run_model_training(input_path_list)