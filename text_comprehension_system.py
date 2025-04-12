
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import openai
import os
from dotenv import load_dotenv

class TextComprehensionSystem:
    def __init__(self):
        load_dotenv()  # Load environment variables
        openai.api_key = os.getenv('OPENAI_API_KEY')

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
            print(f"Web search error: {str(e)}")
            return [f"Web search failed: {str(e)}"]

    def generate_procedure(self, topic: str, input_text: str) -> List[Dict]:
        try:
            prompt = f"""
            Create a detailed work procedure for: {topic}

            Based on this information:
            {input_text}

            Format the response as a list of steps, each containing:
            - Step number
            - Description
            - Requirements (if any)
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional procedure writer."},
                    {"role": "user", "content": prompt}
                ]
            )

            procedure_text = response.choices[0].message.content
            steps = []
            current_step = None

            for line in procedure_text.split('\n'):
                line = line.strip()
                if line.startswith('Step') or line.startswith('#'):
                    if current_step:
                        steps.append(current_step)
                    current_step = {
                        'step': len(steps) + 1,
                        'description': '',
                        'requirements': []
                    }
                elif line.startswith('Requirements:'):
                    if current_step:
                        reqs = line.replace('Requirements:', '').strip()
                        current_step['requirements'] = [r.strip() for r in reqs.split(',')]
                elif line and current_step:
                    if current_step['description']:
                        current_step['description'] += ' ' + line
                    else:
                        current_step['description'] = line

            if current_step:
                steps.append(current_step)

            return steps
        except Exception as e:
            print(f"Procedure generation error: {str(e)}")
            return [{'step': 1, 'description': f"Error generating procedure: {str(e)}", 'requirements': []}]

