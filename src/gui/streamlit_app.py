import streamlit as st
from src.core.model import ModelManager
from src.utils.security import SecurityManager, secure_endpoint
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

class SecureStreamlitApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.security_manager = SecurityManager()
        self.setup_security()
        self.setup_session()

    def setup_security(self):
        """Setup security for the Streamlit app"""
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'failed_attempts' not in st.session_state:
            st.session_state.failed_attempts = 0

    def setup_session(self):
        """Setup session management"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = secrets.token_hex(16)
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = datetime.now()

    def check_session_timeout(self):
        """Check if session has timed out (30 minutes)"""
        if 'last_activity' in st.session_state:
            timeout = timedelta(minutes=30)
            if datetime.now() - st.session_state.last_activity > timeout:
                self.logout()
                return True
        st.session_state.last_activity = datetime.now()
        return False

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user"""
        try:
            # Rate limiting check
            self.security_manager.rate_limit_check(st.request.remote_addr)
            
            # Validate credentials
            if self.security_manager.verify_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.failed_attempts = 0
                return True
            
            # Handle failed attempt
            st.session_state.failed_attempts += 1
            if st.session_state.failed_attempts >= 3:
                self.security_manager.block_ip(st.request.remote_addr)
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False

    def logout(self):
        """Logout user and clear session state"""
        st.session_state.authenticated = False
        st.session_state.failed_attempts = 0
        st.session_state.clear()

    def run(self):
        """Run the secure Streamlit app"""
        if self.check_session_timeout():
            st.warning("Session expired. Please login again.")
            self.show_login_page()
            return
        
        if not st.session_state.authenticated:
            self.show_login_page()
            return

        # Add logout button to sidebar
        if st.sidebar.button("Logout"):
            self.logout()
            st.experimental_rerun()
            return

        # Main app content
        st.title("🔒 Secure AI Procedure Writer")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Generate", "Train Model"])
        
        # Tab 1: Generate content
        with tab1:
            st.header("Generate Procedures")
            # Existing generation code...
            if st.button("Generate Content"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Update progress
                    progress_bar.progress(25)
                    status_text.text("Validating input...")
                    
                    input_text = st.text_area("Enter prompt:")
                    sanitized_input = self.security_manager.sanitize_input(input_text)
                    
                    progress_bar.progress(50)
                    status_text.text("Verifying model integrity...")
                    
                    if self.model_manager.verify_model_integrity():
                        progress_bar.progress(75)
                        status_text.text("Generating content...")
                        
                        result = self.model_manager.generate(sanitized_input)
                        
                        progress_bar.progress(100)
                        status_text.text("Generation complete!")
                        
                        st.write(result)
                    else:
                        st.error("Model integrity check failed")
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

        # Tab 2: Train model
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

                        success = self.model_manager.train_model(
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

    def show_login_page(self):
        """Show secure login page"""
        st.title("🔒 Login Required")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if self.authenticate(username, password):
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

if __name__ == "__main__":
    app = SecureStreamlitApp()
    app.run()





