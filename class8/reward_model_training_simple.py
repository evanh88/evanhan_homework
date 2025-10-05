from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
MODEL_NAME = "microsoft/deberta-v3-base"
DATA_FILE = "reward_data_summaries.jsonl"

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.float32,
    use_safetensors=True,
)

model.to(device)
model.config.use_cache = False

# --- Load dataset ---
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
# Limit dataset size for testing
# dataset = dataset.select(range(min(5, len(dataset))))
print(f"Dataset loaded with {len(dataset)} examples")

# --- Custom Data Collator for Reward Model ---
class RewardDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # Extract chosen and rejected texts
        chosen_texts = [f["chosen"] for f in features]
        rejected_texts = [f["rejected"] for f in features]
        
        # Tokenize both
        chosen_encodings = self.tokenizer(
            chosen_texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            rejected_texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
        }

# --- Custom Reward Model Trainer ---
class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation for reward model"""
        # Get model outputs for chosen and rejected
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )
        
        # Extract logits
        chosen_logits = chosen_outputs.logits.squeeze(-1)
        rejected_logits = rejected_outputs.logits.squeeze(-1)
        
        # Compute reward loss: we want chosen > rejected
        # Use sigmoid to ensure stability
        loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits)).mean()
        
        if return_outputs:
            return loss, (chosen_outputs, rejected_outputs)
        return loss

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    fp16=False,
    bf16=False,
    logging_steps=1,
    save_strategy="no",
    remove_unused_columns=False,
    report_to="none",
    dataloader_drop_last=True,
    max_grad_norm=1.0,
    warmup_steps=0,
    save_steps=1000,
    eval_strategy="no",
)

# --- Data Collator ---
data_collator = RewardDataCollator(tokenizer)

# --- Trainer ---
trainer = RewardModelTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# --- Train ---
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    raise e

# --- Save model ---
trainer.save_model()
tokenizer.save_pretrained("./reward_model")
print("Model saved successfully!")
