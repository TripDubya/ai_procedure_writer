import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
from datetime import datetime
import torch
import tempfile
import logging
import sys
from streamlit.web.server import Server
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Streamlit's file watcher for PyTorch
Server.file_watcher_type = "none"

# Initialize model and tokenizer outside of the main function
@st.cache_resource
def initialize_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        if torch.cuda.is_available():
            model = model.to('cuda')
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return None, None

def get_model():
    """Get or create model instance"""
    return initialize_model()

def generate_procedure(prompt, model, tokenizer, max_length=200):
    """Generate procedure text using the model"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        logger.error(f"Error generating procedure: {str(e)}")
        return None

def save_procedure(prompt, procedure):
    """Save the generated procedure to a file"""
    try:
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
    except Exception as e:
        logger.error(f"Error saving procedure: {str(e)}")
        return None

def train_model(model, tokenizer, training_text, epochs=1):
    """Train the model on new data"""
    temp_file = None
    try:
        # Create temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write(training_text)
            temp_file = f.name

        # Prepare dataset
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=temp_file,
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./training_output",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10000,
            save_total_limit=2,
            logging_steps=100,
            logging_dir="./logs"
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Train the model
        trainer.train()
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False
    
    finally:
        # Cleanup
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

def main():
    try:
        # Disable file watching for this session
        ctx = get_script_run_ctx()
        if ctx is not None:
            ctx.file_watcher_type = "none"
            
        st.set_page_config(page_title="AI Procedure Writer", page_icon="📝", layout="wide")
        
        st.title("📝 AI Procedure Writer")
        st.write("Generate and train AI to write procedures")

        # Initialize model
        model, tokenizer = get_model()
        if model is None or tokenizer is None:
            st.error("Failed to initialize the application. Please refresh the page to try again.")
            return

        # Create tabs for different functions
        tab1, tab2 = st.tabs(["Generate Procedures", "Train AI"])

        # Tab 1: Generate Procedures
        with tab1:
            st.header("Generate New Procedure")
            
            prompt = st.text_area(
                "Enter what procedure you want to generate:", 
                height=100,
                placeholder="Example: How to make a perfect cup of coffee"
            )

            with st.expander("Advanced Options"):
                max_length = st.slider("Maximum length", 100, 500, 200)

            if st.button("Generate Procedure", type="primary"):
                if not prompt:
                    st.error("Please enter a prompt first!")
                    return
                    
                with st.spinner('Generating procedure...'):
                    procedure = generate_procedure(
                        prompt, 
                        st.session_state['model'],
                        st.session_state['tokenizer'],
                        max_length=max_length
                    )
                    
                    if procedure:
                        filename = save_procedure(prompt, procedure)
                        if filename:
                            st.success("Procedure generated successfully!")
                            st.write("### Generated Procedure:")
                            st.write(procedure)
                            
                            with open(filename, 'r', encoding='utf-8') as f:
                                st.download_button(
                                    label="Download Procedure",
                                    data=f.read(),
                                    file_name=os.path.basename(filename),
                                    mime="text/plain"
                                )
                        else:
                            st.error("Failed to save the generated procedure.")
                    else:
                        st.error("Failed to generate procedure. Please try again.")

        # Tab 2: Train AI
        with tab2:
            st.header("Train AI with New Data")
            
            uploaded_file = st.file_uploader(
                "Upload training data (TXT file)", 
                type=['txt'],
                help="Upload a text file containing examples of procedures"
            )
            
            training_text = st.text_area(
                "Or paste training text directly:",
                height=200,
                placeholder="Paste your training text here..."
            )

            with st.expander("Training Options"):
                epochs = st.slider("Training epochs", 1, 10, 1)

            if st.button("Train AI", type="primary"):
                if not (uploaded_file or training_text):
                    st.error("Please provide training data first!")
                    return
                    
                with st.spinner('Training AI... This may take a while...'):
                    try:
                        training_data = (uploaded_file.getvalue().decode('utf-8') 
                                       if uploaded_file is not None 
                                       else training_text)

                        success = train_model(
                            st.session_state['model'],
                            st.session_state['tokenizer'],
                            training_data,
                            epochs=epochs
                        )
                        
                        if success:
                            st.success("Training completed successfully!")
                        else:
                            st.error("Training failed. Please try again.")
                    except Exception as e:
                        logger.error(f"Training failed: {str(e)}")
                        st.error(f"Training failed: {str(e)}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()


