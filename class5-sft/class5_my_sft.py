import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import Trainer, DataCollatorForLanguageModeling

# -----------------------------
# 1. Load Tokenizer
# -----------------------------
model_name = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 3. Load Dataset
# -----------------------------
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# -----------------------------
# 4. Prompt Formatter
# -----------------------------
def format_sample(example):
    instruction = example["text"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()

    if input_text:
        prompt = f"### Human: {instruction}\n{input_text}\n### Assistant: "
    else:
        prompt = f"### Human: {instruction}\n### Assistant: "

    full_text = prompt + output
    return {"prompt": prompt, "response": output, "full_text": full_text}

# -----------------------------
# 5. Tokenization + Label Masking
# -----------------------------
def tokenize(example):
    formatted = format_sample(example)
    full_text = formatted["full_text"]
    prompt_text = formatted["prompt"]

    tokenized_full = tokenizer(full_text, truncation=True, max_length=1024, padding="max_length")
    tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=1024)

    labels = tokenized_full["input_ids"].copy()
    labels[:len(tokenized_prompt["input_ids"])] = [-100] * len(tokenized_prompt["input_ids"])
    tokenized_full["labels"] = labels
    return tokenized_full

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# -----------------------------
# 6. Data Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# 2. Load model adn propare for LoRA
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             trust_remote_code=True)

model.config.use_cache = False  # Needed for gradient checkpointing

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "q_proj", "v_proj", "k_proj", "o_proj"],  # adjust if needed
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 7. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./qwen2.5-3b-lora-guanaco",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="tensorboard",
    remove_unused_columns=False,
    logging_first_step=True
)

# -----------------------------
# 8. Trainer Setup
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -----------------------------
# 9. Train
# -----------------------------
trainer.train()

# -----------------------------
# 10. Save Adapter
# -----------------------------
model.save_pretrained("./qwen2.5-3b-lora-guanaco")
tokenizer.save_pretrained("./qwen2.5-3b-lora-guanaco")
