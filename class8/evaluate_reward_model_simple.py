from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluate import load
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
# rouge_score
# bert_score

# Load evaluation metrics
rouge = load("rouge")
bertscore = load("bertscore")

# --- Load your trained reward model ---
MODEL_PATH = "./reward_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
 
# --- Load dataset ---
dataset = load_dataset("json", data_files="reward_data_summaries.jsonl", split="train")
print(f"Loaded dataset with {len(dataset)} examples")

# --- Function to get reward model scores ---
def get_reward_score(text):
    """Get reward score from the trained model"""
    with torch.no_grad():
        inputs = tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        score = outputs.logits.squeeze(-1).item()
        return score

# --- Evaluate on a subset ---
print("Computing metrics...")
results = []

# Use first 20 examples for evaluation
eval_dataset = dataset.select(range(min(20, len(dataset))))

for i, example in enumerate(eval_dataset):
    print(f"Processing example {i+1}/{len(eval_dataset)}")
    
    chosen = example['chosen']
    rejected = example['rejected']
    
    # Get reward model scores
    chosen_reward = get_reward_score(chosen)
    rejected_reward = get_reward_score(rejected)
    
    # Use rejected as reference (common approach when no gold reference)
    reference = rejected
    
    # Compute ROUGE scores
    rouge_chosen = rouge.compute(predictions=[chosen], references=[reference])
    rouge_rejected = rouge.compute(predictions=[rejected], references=[reference])
    
    # Compute BERTScore
    bertscore_chosen = bertscore.compute(predictions=[chosen], references=[reference], lang="en")
    bertscore_rejected = bertscore.compute(predictions=[rejected], references=[reference], lang="en")
    
    # Store results
    results.append({
        'chosen_reward': chosen_reward,
        'rejected_reward': rejected_reward,
        'reward_diff': chosen_reward - rejected_reward,
        'chosen_rouge_1': rouge_chosen['rouge1'],
        'rejected_rouge_1': rouge_rejected['rouge1'],
        'rouge_1_diff': rouge_chosen['rouge1'] - rouge_rejected['rouge1'],
        'chosen_rouge_2': rouge_chosen['rouge2'],
        'rejected_rouge_2': rouge_rejected['rouge2'],
        'rouge_2_diff': rouge_chosen['rouge2'] - rouge_rejected['rouge2'],
        'chosen_rouge_l': rouge_chosen['rougeL'],
        'rejected_rouge_l': rouge_rejected['rougeL'],
        'rouge_l_diff': rouge_chosen['rougeL'] - rouge_rejected['rougeL'],
        'chosen_bertscore_f1': bertscore_chosen['f1'][0],
        'rejected_bertscore_f1': bertscore_rejected['f1'][0],
        'bertscore_f1_diff': bertscore_chosen['f1'][0] - bertscore_rejected['f1'][0],
    })

# --- Compute correlations ---
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

reward_diffs = [r['reward_diff'] for r in results]
rouge_1_diffs = [r['rouge_1_diff'] for r in results]
rouge_2_diffs = [r['rouge_2_diff'] for r in results]
rouge_l_diffs = [r['rouge_l_diff'] for r in results]
bertscore_f1_diffs = [r['bertscore_f1_diff'] for r in results]

# Correlations
rouge_1_spearman = spearmanr(reward_diffs, rouge_1_diffs)[0]
rouge_1_pearson = pearsonr(reward_diffs, rouge_1_diffs)[0]

rouge_2_spearman = spearmanr(reward_diffs, rouge_2_diffs)[0]
rouge_2_pearson = pearsonr(reward_diffs, rouge_2_diffs)[0]

rouge_l_spearman = spearmanr(reward_diffs, rouge_l_diffs)[0]
rouge_l_pearson = pearsonr(reward_diffs, rouge_l_diffs)[0]

bertscore_spearman = spearmanr(reward_diffs, bertscore_f1_diffs)[0]
bertscore_pearson = pearsonr(reward_diffs, bertscore_f1_diffs)[0]

print(f"ROUGE-1 vs Reward Model:")
print(f"  Spearman correlation: {rouge_1_spearman:.3f}")
print(f"  Pearson correlation: {rouge_1_pearson:.3f}")

print(f"\nROUGE-2 vs Reward Model:")
print(f"  Spearman correlation: {rouge_2_spearman:.3f}")
print(f"  Pearson correlation: {rouge_2_pearson:.3f}")

print(f"\nROUGE-L vs Reward Model:")
print(f"  Spearman correlation: {rouge_l_spearman:.3f}")
print(f"  Pearson correlation: {rouge_l_pearson:.3f}")

print(f"\nBERTScore-F1 vs Reward Model:")
print(f"  Spearman correlation: {bertscore_spearman:.3f}")
print(f"  Pearson correlation: {bertscore_pearson:.3f}")

# --- Ranking agreement ---
print("\n" + "="*50)
print("RANKING AGREEMENT")
print("="*50)

# Count how many times each metric agrees with reward model ranking
reward_rankings = [1 if r['reward_diff'] > 0 else 0 for r in results]
rouge_1_rankings = [1 if r['rouge_1_diff'] > 0 else 0 for r in results]
rouge_2_rankings = [1 if r['rouge_2_diff'] > 0 else 0 for r in results]
rouge_l_rankings = [1 if r['rouge_l_diff'] > 0 else 0 for r in results]
bertscore_rankings = [1 if r['bertscore_f1_diff'] > 0 else 0 for r in results]

rouge_1_agreement = sum(1 for i in range(len(results)) if reward_rankings[i] == rouge_1_rankings[i]) / len(results)
rouge_2_agreement = sum(1 for i in range(len(results)) if reward_rankings[i] == rouge_2_rankings[i]) / len(results)
rouge_l_agreement = sum(1 for i in range(len(results)) if reward_rankings[i] == rouge_l_rankings[i]) / len(results)
bertscore_agreement = sum(1 for i in range(len(results)) if reward_rankings[i] == bertscore_rankings[i]) / len(results)

print(f"ROUGE-1 agreement: {rouge_1_agreement:.3f}")
print(f"ROUGE-2 agreement: {rouge_2_agreement:.3f}")
print(f"ROUGE-L agreement: {rouge_l_agreement:.3f}")
print(f"BERTScore agreement: {bertscore_agreement:.3f}")

# --- Summary ---
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total examples: {len(results)}")
print(f"Reward model accuracy: {sum(reward_rankings)/len(reward_rankings):.3f}")
print(f"Average reward difference: {np.mean(reward_diffs):.3f}")

print("\nEvaluation completed!")
