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
import json
import argparse
import numpy as np

# -----------------------------
# 1. Load Tokenizer
# -----------------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--finetune_mode",
    choices=["lora", "full"],
    default="lora",
    help="Choose LoRA (QLoRA) or full fine-tuning"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs"
)
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="Learning rate override. If not set, a sensible default per mode is used."
)
args = parser.parse_args()

bnb_config = None
if args.finetune_mode == "lora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 2. Load Dataset
# -----------------------------
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(1000))
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# -----------------------------
# 3. Prompt Formatter
# -----------------------------
system_token = "<|im_start|>system"
user_token = "<|im_start|>user"
assistant_token = "<|im_start|>assistant"
end_token = "<|im_end|>"

def format_segments(example):
    """Return list of (text, role) segments in Qwen chat format for precise label masking."""
    conversation_text = example['text']
    raw_segments = conversation_text.split('###')

    segments = []
    # system
    segments.append((f"{system_token}\nYou are a helpful assistant.{end_token}\n", "system"))

    # Iterate over pairs of segments
    for i in range(1, len(raw_segments) - 1, 2):
        human_text = raw_segments[i].strip().replace('Human:', '').strip()
        if i + 1 < len(raw_segments):
            assistant_text = raw_segments[i+1].strip().replace('Assistant:', '').strip()
            segments.append((f"{user_token}\n{human_text}{end_token}\n", "user"))
            segments.append((f"{assistant_token}\n{assistant_text}{end_token}\n", "assistant"))
        else:
            segments.append((f"{user_token}\n{human_text}{end_token}\n", "user"))

    return segments

def format_sample(example):
    segments = format_segments(example)
    return "".join([t for (t, _) in segments])

# -----------------------------
# 4. Tokenization + Label Masking
# -----------------------------
def tokenize(example):
    segments = format_segments(example)
    input_ids = []
    labels = []
    attention_mask = []

    for text, role in segments:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        seg_ids = enc["input_ids"]
        input_ids.extend(seg_ids)
        # Mask everything except assistant tokens
        if role == "assistant":
            labels.extend(seg_ids)
        else:
            labels.extend([-100] * len(seg_ids))
        attention_mask.extend([1] * len(seg_ids))

    # Truncate to max_length
    max_length = 4096
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = attention_mask[:max_length]

    # Pad to max_length
    pad_id = tokenizer.pad_token_id
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_train = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)
tokenized_eval = dataset["test"].map(tokenize, remove_columns=dataset["test"].column_names)
print(tokenized_train)
# print(tokenized_dataset['train'])

# -----------------------------
# 5. Data Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# 6. Load model and prepare for selected mode
# -----------------------------
if args.finetune_mode == "lora":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False  # Needed for gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
else:
    # Full fine-tuning: load the full-precision model (bf16 if available)
    preferred_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=preferred_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

# -----------------------------
# 7. Training Arguments
# -----------------------------
default_lr = 2e-4 if args.finetune_mode == "lora" else 1e-5
use_fp16 = True if args.finetune_mode == "lora" else (not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()))

training_args = TrainingArguments(
    output_dir=f"./qwen2.5-1.5b-{args.finetune_mode}-guanaco",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=6,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=default_lr if args.lr is None else args.lr,
    fp16=use_fp16,
    bf16=(not use_fp16),
    save_strategy="epoch",
    save_total_limit=2,
    report_to="tensorboard",
    remove_unused_columns=False,
    logging_first_step=True,
    eval_strategy="steps",
    eval_steps=200,
)

# -----------------------------
# 8. Trainer Setup
# -----------------------------
def compute_metrics(eval_pred):
    """Compute token-level accuracy and simple proxy helpfulness/style metrics.
    Loss is reported by Trainer automatically.
    """
    # Support both tuple and EvalPrediction objects
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    # If predictions are logits (ndarray), take argmax
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if hasattr(predictions, "ndim") and predictions.ndim == 3:
        pred_ids = predictions.argmax(-1)
    else:
        pred_ids = predictions

    # Accuracy on positions where label != -100
    labels_np = np.array(labels)
    pred_np = np.array(pred_ids)
    label_mask = labels_np != -100
    correct = (pred_np == labels_np) & label_mask
    accuracy = correct.sum() / np.maximum(label_mask.sum(), 1)

    # Decode predicted assistant tokens per-sample
    helpful_keywords = [
        "help", "assist", "sure", "certainly", "glad", "happy to",
        "let me", "here is", "here's"
    ]
    style_keywords = [
        "please", "thanks", "thank you", "step-by-step", "let's", "we can"
    ]

    batch_helpful = []
    batch_style = []
    for i in range(labels_np.shape[0]):
        mask_i = label_mask[i]
        if not np.any(mask_i):
            batch_helpful.append(0.0)
            batch_style.append(0.0)
            continue
        pred_tokens = pred_np[i][mask_i]
        try:
            text = tokenizer.decode(pred_tokens, skip_special_tokens=True).lower()
        except Exception:
            text = ""
        helpful_hit = any(kw in text for kw in helpful_keywords)
        style_hit = any(kw in text for kw in style_keywords)
        batch_helpful.append(1.0 if helpful_hit else 0.0)
        batch_style.append(1.0 if style_hit else 0.0)

    metrics = {
        "accuracy": float(accuracy),
        "helpfulness": float(np.mean(batch_helpful) if batch_helpful else 0.0),
        "style_match": float(np.mean(batch_style) if batch_style else 0.0),
    }
    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------------
# 9. Train
# -----------------------------
trainer.train()

# -----------------------------
# 10. Save model
# -----------------------------
save_dir = f"./qwen2.5-1.5b-{args.finetune_mode}-guanaco"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
