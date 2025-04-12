from dotenv import load_dotenv
import os

def test_system():
    # Load environment variables from the correct path
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    print("\nAPI Key verification:", "Available" if api_key else "Not Found")
    if api_key:
        print("API Key length:", len(api_key))
        print("API Key starts with:", api_key[:7])
        print("Full path to .env:", env_path)

if __name__ == "__main__":
    test_system()


