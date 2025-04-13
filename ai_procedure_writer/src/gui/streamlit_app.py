import streamlit as st
from bs4 import BeautifulSoup
import requests
import logging
from typing import List

class SecureStreamlitApp:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def search_web(self, query: str, num_results: int = 5) -> List[str]:
        try:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            for result in soup.find_all('div', class_='result__body')[:num_results]:
                text = result.get_text().strip()
                if text:
                    results.append(text)

            return results
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return [f"Web search failed: {str(e)}"]

    def run(self):
        st.title("🔒 AI Procedure Writer")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Generate", "Train Model", "Web Search Training"])
        
        # Tab 1: Generate content
        with tab1:
            st.header("Generate Procedures")
            input_text = st.text_area("Enter prompt:")
            if st.button("Generate Content"):
                st.write("Generation functionality coming soon...")

        # Tab 2: Train model with file upload
        with tab2:
            st.header("Train AI with Data")
            
            # File upload section
            st.subheader("Upload Training Data")
            uploaded_files = st.file_uploader(
                "Upload training files", 
                type=['txt', 'md', 'csv'], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    file_contents = file.read().decode('utf-8')
                    st.write(f"File: {file.name}")
                    with st.expander("Preview Content"):
                        st.text(file_contents[:500] + "...")

        # Tab 3: Web Search Training
        with tab3:
            st.header("Web Search for Training Data")
            
            search_query = st.text_input("Enter search query:")
            num_results = st.slider("Number of results", 1, 20, 5)
            
            if st.button("Search Web"):
                with st.spinner('Searching...'):
                    results = self.search_web(search_query, num_results)
                    
                    if results:
                        st.success("Search completed!")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}"):
                                st.write(result)
                    else:
                        st.warning("No results found")

if __name__ == "__main__":
    app = SecureStreamlitApp()
    app.run()