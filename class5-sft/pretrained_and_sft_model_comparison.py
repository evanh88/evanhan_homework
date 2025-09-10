import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import json

class ModelComparison:
    def __init__(self):
        self.base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.fine_tuned_model_path = "./qwen2.5-1.5b-lora-guanaco"
        
        # Configure quantization for memory efficiency
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize models
        self.base_model = None
        self.fine_tuned_model = None
        
    def load_base_model(self):
        """Load the pre-trained Qwen2.5-1.5B-Instruct model"""
        print("Loading pre-trained Qwen2.5-1.5B-Instruct model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Base model loaded successfully!")
        
    def load_fine_tuned_model(self):
        """Load the fine-tuned model with LoRA adapter"""
        print("Loading fine-tuned qwen2.5-1.5b-lora-guanaco model...")
        
        # First load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Then load the LoRA adapter
        self.fine_tuned_model = PeftModel.from_pretrained(
            base_model,
            self.fine_tuned_model_path,
            device_map="auto"
        )
        print("✓ Fine-tuned model loaded successfully!")
        
    def format_prompt(self, user_query, system_message="You are a helpful assistant."):
        """Format the prompt in Qwen2.5 chat format"""
        system_token = "<|im_start|>system"
        user_token = "<|im_start|>user"
        assistant_token = "<|im_start|>assistant"
        end_token = "<|im_end|>"
        
        formatted_prompt = f"{system_token}\n{system_message}{end_token}\n"
        formatted_prompt += f"{user_token}\n{user_query}{end_token}\n"
        formatted_prompt += f"{assistant_token}\n"
        
        return formatted_prompt
    
    def generate_answer(self, model, user_query, max_length=512, temperature=0.7, top_p=0.9):
        """Generate answer from a model"""
        # Format the prompt
        prompt = self.format_prompt(user_query)
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_start = full_response.find("\nassistant\n")
        if assistant_start != -1:
            assistant_response = full_response[assistant_start + len("<\nassistant"):]
            # Remove any trailing end tokens
            # assistant_response = assistant_response.replace("<|im_end|>", "").strip()
        else:
            assistant_response = full_response[len(prompt):].strip()
        
        return assistant_response
    
    def compare_models(self, user_query, max_length=512, temperature=0.7, top_p=0.9):
        """Compare responses from both models"""
        print(f"\n{'='*80}")
        print(f"QUERY: {user_query}")
        print(f"{'='*80}")
        
        results = {}
        
        # Generate answer from base model
        if self.base_model is not None:
            print("\n🤖 PRE-TRAINED MODEL (Qwen2.5-1.5B-Instruct):")
            print("-" * 50)
            try:
                base_answer = self.generate_answer(
                    self.base_model, user_query, max_length, temperature, top_p
                )
                print(base_answer)
                results['base_model'] = base_answer
            except Exception as e:
                print(f"Error generating answer from base model: {e}")
                results['base_model'] = f"Error: {e}"
        else:
            print("Base model not loaded!")
            results['base_model'] = "Model not loaded"
        
        # Generate answer from fine-tuned model
        if self.fine_tuned_model is not None:
            print("\n🎯 FINE-TUNED MODEL (qwen2.5-1.5b-lora-guanaco):")
            print("-" * 50)
            try:
                fine_tuned_answer = self.generate_answer(
                    self.fine_tuned_model, user_query, max_length, temperature, top_p
                )
                print(fine_tuned_answer)
                results['fine_tuned_model'] = fine_tuned_answer
            except Exception as e:
                print(f"Error generating answer from fine-tuned model: {e}")
                results['fine_tuned_model'] = f"Error: {e}"
        else:
            print("Fine-tuned model not loaded!")
            results['fine_tuned_model'] = "Model not loaded"
        
        return results
    
    def run_example_queries(self):
        """Run a set of example queries to compare both models"""
        example_queries = [
        "Which open-source packages are good for simulations of chemical unit operations using Python?",
        "How to optimize my webpage for search engines??",
        "What is a black swan",
        "What do you eat?",
        "Write a Python function to read a jsonl file."
        ]
        
        all_results = {}
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n\n🔍 EXAMPLE {i}/{len(example_queries)}")
            results = self.compare_models(query)
            all_results[f"query_{i}"] = {
                "query": query,
                "base_model_response": results.get('base_model', ''),
                "fine_tuned_model_response": results.get('fine_tuned_model', '')
            }
        
        # Save results to file
        with open("model_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n\n📊 All results saved to 'model_comparison_results.json'")
        return all_results

def main():
    """Main function to run the model comparison"""
    print("🚀 Starting Model Comparison Script")
    print("=" * 50)
    
    # Initialize the comparison class
    comparator = ModelComparison()
    
    # Load both models
    try:
        comparator.load_base_model()
        comparator.load_fine_tuned_model()
        
        print("\n✅ Both models loaded successfully!")
        
        # Run example queries
        print("\n🎯 Running example queries...")
        results = comparator.run_example_queries()
        
        print("\n🎉 Model comparison completed!")
        print("Check 'model_comparison_results.json' for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during model loading or comparison: {e}")
        print("Please check that:")
        print("1. The base model name is correct")
        print("2. The fine-tuned model path exists")
        print("3. You have sufficient GPU memory")
        print("4. All required packages are installed")

if __name__ == "__main__":
    main()
