from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

# Define some test questions (ensure these were not exactly in training data)
test_questions = [
    "What is fine tuning of AI models?",
    "What problem does FlowRL address in existing reasoning models?",
    "Can you tell me about the methods in AI-driven diagnostics?",
    "What is the key principle behind the EVOL-RL approach?",
    "What is the proposed solution to the long decoding-window problem?",
    "What are the two stages of the MEDFACT-R1 framework?",
    "What machine learning techniques are available currently",
    "What are the advantages of the Super-Linear model?",
    "What challenge does reinforcement learning face in non-stationary environments?",
    "What is the main purpose of the AdaMM framework"
]

system_prompt = "You are a research assistant specialized in academic research"

BOT = "<|begin_of_text|>"
EOT = "<|end_of_text|>"
SOF = "<|start_header_id|>"
EOF = "<|end_header_id|>"
EOM = "<|eom_id|>"  # End of message, for multi-turn conversations
EOTN = "<|eot_id|>\n"  # End of turn

model_name = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"

# Function to generate response
def generate_response(model, tokenizer, question, system_prompt):
    prompt_input = BOT + SOF + "system" + EOF + system_prompt + EOTN
    prompt_input += SOF + "user" + EOF + question + EOTN
    prompt_input += SOF + "assistant" + EOF + "\n"
    input_ids = tokenizer(prompt_input, return_tensors='pt').input_ids.cuda()
    output_ids = model.generate(input_ids, max_new_tokens=150)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Post-process to remove the prompt part
    answer = answer.split('assistant\n')[-1].strip()
    return answer

# Load the base model and perform inference
print("Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("Generating responses from base model...")
for que in test_questions:
    base_answer = generate_response(base_model, base_tokenizer, que, system_prompt)
    print(f"Q: {que}")
    print(f"Base Model Answer: {base_answer}")
    print("-" * 60)

# Clear GPU memory before loading the fine-tuned model
torch.cuda.empty_cache()
del base_model
del base_tokenizer
torch.cuda.empty_cache()


# Load the fine-tuned model and perform inference
print("Loading fine-tuned model...")
"""
# Load the base model first
ft_model, _ = FastLanguageModel.from_pretrained(
    model_name, # Load base model
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Load the tokenizer from the base model name
ft_tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load the adapter weights from the checkpoint
ft_model.load_adapter("llama3-8b-qlora-finetuned")
"""

# Load base model + adapter in one step
ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.1-8b-unsloth-bnb-4bit",  # or your base model
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    adapter_path = "llama3-8b-qlora-finetuned",  # ✅ Your fine-tuned adapter folder
    token = False  # Optional: if model is gated on Hugging Face
)

print("Generating responses from fine-tuned model...")
for que in test_questions:
    ft_answer = generate_response(ft_model, ft_tokenizer, que, system_prompt)
    print(f"Q: {que}")
    print(f"Fine-Tuned Model Answer: {ft_answer}")
    print("-" * 60)

# Clear GPU memory after inference
del ft_model
del ft_tokenizer
torch.cuda.empty_cache()