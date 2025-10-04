from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
import torch

# prefer not to rely on the collator at all
# from torch.utils.data import default_collate

# def collate_fn(examples):
#     return default_collate(examples)

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
    torch_dtype=torch.float32,  # Use consistent dtype
    use_safetensors=True,
)

model.to(device)

# Configure model for training
model.config.use_cache = False
model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency

# --- Load dataset ---
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"Dataset loaded with {len(dataset)} examples")
print(f"Sample: {dataset[0]}")

# --- RewardTrainer config ---
training_args = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size to prevent memory issues
    gradient_accumulation_steps=8,  # Increase accumulation to maintain effective batch size
    learning_rate=1e-5,
    fp16=False,  # Disable fp16 to avoid dtype conflicts
    bf16=True if torch.cuda.is_available() else False,  # Use bf16 if available
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none",
    disable_dropout=False,
    dataloader_drop_last=True,  # Drop incomplete batches
    max_grad_norm=1.0,  # Add gradient clipping
    warmup_steps=100,  # Add warmup
)

# --- Trainer ---
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    # data_collator=collate_fn  # required for TRL ≥ 0.10
)

# --- Train ---
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    # Clear cache and try again with different settings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    raise e

# --- Save model ---
trainer.save_model()
tokenizer.save_pretrained("./reward_model")
print("Model saved successfully!")
