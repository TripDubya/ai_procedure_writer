import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from datetime import datetime

def initialize_model():
    """Initialize the GPT-2 model and tokenizer"""
    with st.spinner('Loading AI model... (this may take a minute)'):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model, tokenizer

def generate_procedure(prompt, model, tokenizer, max_length=200):
    """Generate procedure text using the model"""
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def save_procedure(prompt, procedure):
    """Save the generated procedure to a file"""
    output_dir = "generated_procedures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/procedure_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Generated Procedure:\n")
        f.write(procedure)
    
    return filename

def main():
    st.set_page_config(page_title="AI Procedure Writer", page_icon="📝")
    
    st.title("📝 AI Procedure Writer")
    st.write("Generate step-by-step procedures using AI")

    # Initialize model (only once)
    if 'model' not in st.session_state:
        model, tokenizer = initialize_model()
        st.session_state['model'] = model
        st.session_state['tokenizer'] = tokenizer

    # Input area
    prompt = st.text_area("Enter what procedure you want to generate:", 
                         height=100,
                         placeholder="Example: How to make a perfect cup of coffee")

    # Generation options
    with st.expander("Advanced Options"):
        max_length = st.slider("Maximum length", 100, 500, 200)

    # Generate button
    if st.button("Generate Procedure", type="primary"):
        if prompt:
            with st.spinner('Generating procedure...'):
                procedure = generate_procedure(prompt, 
                                            st.session_state['model'],
                                            st.session_state['tokenizer'],
                                            max_length=max_length)
                filename = save_procedure(prompt, procedure)
            
            # Display results
            st.success("Procedure generated successfully!")
            st.write("### Generated Procedure:")
            st.write(procedure)
            
            # Download button
            with open(filename, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="Download Procedure",
                    data=f.read(),
                    file_name=os.path.basename(filename),
                    mime="text/plain"
                )
        else:
            st.error("Please enter a prompt first!")

    # Help section
    with st.expander("Help & Tips"):
        st.markdown("""
        ### How to use:
        1. Enter what procedure you want to generate in the text box
        2. Click the 'Generate Procedure' button
        3. View the generated procedure
        4. Download the procedure if you want to save it
        
        ### Tips for good prompts:
        - Be specific about what you want
        - Include the context or purpose
        - Mention if you need specific types of steps
        
        ### Example prompts:
        - "How to backup important files on a computer"
        - "Steps to organize a small team meeting"
        - "Procedure for setting up a new workspace"
        """)

if __name__ == "__main__":
    main()