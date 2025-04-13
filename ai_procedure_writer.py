from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from datetime import datetime

def initialize_model():
    """Initialize the GPT-2 model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model, tokenizer

def generate_procedure(prompt, model, tokenizer, max_length=200):
    """Generate procedure text using the model"""
    # Prepare input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def save_procedure(prompt, procedure):
    """Save the generated procedure to a file"""
    # Create directory if it doesn't exist
    output_dir = "generated_procedures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/procedure_{timestamp}.txt"
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Generated Procedure:\n")
        f.write(procedure)
    
    return filename

def main():
    print("=== AI Procedure Writer ===")
    
    # Initialize model
    print("Initializing AI model...")
    model, tokenizer = initialize_model()
    
    while True:
        # Get user input
        print("\nEnter your prompt for generating the work procedure")
        print("(or 'quit' to exit):")
        prompt = input("> ")
        
        if prompt.lower() == 'quit':
            break
        
        # Generate procedure
        print("\nGenerating procedure...")
        procedure = generate_procedure(prompt, model, tokenizer)
        
        # Save and display
        filename = save_procedure(prompt, procedure)
        print("\nGenerated Procedure:")
        print("-" * 40)
        print(procedure)
        print("-" * 40)
        print(f"\nProcedure saved to: {filename}")

if __name__ == "__main__":
    main()
