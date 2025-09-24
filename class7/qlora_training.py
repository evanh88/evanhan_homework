from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
import os
import wandb

wandb.init (mode="disabled")

# Load the base LLaMA 3 7B model in 4-bit mode (dynamic 4-bit quantization)
model_name = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained( # Unpack the tuple
    model_name,
    max_seq_length = 2048, # Add a default max_seq_length
    dtype = None, # Auto detect the dtype
    load_in_4bit = True, # Loads in 4 bit
)

# Configure the model with PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load our synthetic Q&A dataset
dataset = load_dataset("json", data_files="synthetic_qa.jsonl", split="train")

# Initialize the trainer for Supervised Fine-Tuning (SFT)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        output_dir="llama3-8b-qlora-finetuned",
        per_device_train_batch_size=2,   # small batch size for Colab GPU
        gradient_accumulation_steps=8,   # accumulate gradients to simulate larger batch
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True, # Set fp16 to False to avoid dtype mismatch
        logging_steps=50,
        save_strategy="epoch"
        save_total_limit=1
    )
)

trainer.train()
model.save_pretrained("llama3-8b-qlora-finetuned")
# For comparison of file struction of saved model
model.save_model("llama3-8b-qlora-finetuned_model")

# Optional: Merge LoRA weights into the base model for inference
# model = model.merge_and_unload()
# model.save_pretrained("path/to/merged_model")