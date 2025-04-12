import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime
import logging
import os

# Initialize logging configuration
logging.basicConfig(
    filename='ai_procedure_writer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AIProcedureWriter:
    def __init__(self):
        try:
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            logging.info("Model and tokenizer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            raise
        
    def generate_procedure(self, prompt):
        try:
            # Input validation
            self.input_validation(prompt)
            
            # Tokenize and generate
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            output = self.model.generate(
                input_ids,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.92,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_procedure = self.tokenizer.decode(output[0], skip_special_tokens=True)
            logging.info(f"Successfully generated procedure for prompt: {prompt[:50]}...")
            return generated_procedure
            
        except Exception as e:
            logging.error(f"Error occurred in procedure generation: {str(e)}")
            return None

    def monitor_model_performance(self):
        try:
            # Basic performance monitoring
            memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
            logging.info(f"Current GPU memory usage: {memory_usage}")
            return memory_usage
        except Exception as e:
            logging.error(f"Error in performance monitoring: {str(e)}")
        
    def model_optimization(self):
        try:
            # Basic model optimization
            if torch.cuda.is_available():
                self.model.to('cuda')
                logging.info("Model moved to GPU for optimization")
            torch.backends.cudnn.benchmark = True
            logging.info("CUDNN benchmark enabled for optimization")
        except Exception as e:
            logging.error(f"Error in model optimization: {str(e)}")

    def input_validation(self, prompt):
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > 1000:
            raise ValueError("Prompt is too long (max 1000 characters)")

    def save_procedure(self, generated_text):
        try:
            output_dir = "generated_procedures"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.join(
                output_dir,
                f"procedure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Generated on: {datetime.now()}\n")
                f.write("-" * 50 + "\n")
                f.write(generated_text)
                
            logging.info(f"Procedure saved to {filename}")
            return filename
        except Exception as e:
            logging.error(f"Error saving procedure: {str(e)}")
            return None

    def user_interface(self):
        print("\n=== AI Procedure Writer ===")
        print("Enter your prompt for generating the work procedure")
        print("(or 'quit' to exit):")
        return input("> ").strip()

def main():
    ai_writer = AIProcedureWriter()
    
    while True:
        try:
            prompt = ai_writer.user_interface()
            
            if prompt.lower() == 'quit':
                print("Goodbye!")
                break
                
            generated_text = ai_writer.generate_procedure(prompt)
            
            if generated_text:
                print("\nGenerated Procedure:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
                
                saved_file = ai_writer.save_procedure(generated_text)
                if saved_file:
                    print(f"\nProcedure saved to: {saved_file}")
                
                ai_writer.monitor_model_performance()
                ai_writer.model_optimization()
            else:
                print("\nFailed to generate procedure. Please try again.")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()