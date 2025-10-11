import os
import requests

def llm_pipeline(prompt, max_new_tokens=150):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",  # ✅ valid model
        "messages": [
            {"role": "system", "content": "You are an expert dissertation assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_completion_tokens": max_new_tokens  # ✅ correct field name per docs
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    return [{"generated_text": data["choices"][0]["message"]["content"]}]
