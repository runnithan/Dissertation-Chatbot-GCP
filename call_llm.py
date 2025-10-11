import os
import requests

def llm_pipeline(prompt, max_new_tokens=150):
    """
    Calls the Groq API to generate a dissertation-style response.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mixtral-8x7b-32768",  # Groq’s most stable instruction-tuned model
        "messages": [
            {"role": "system", "content": "You are an expert dissertation assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_new_tokens,
    }

    # ✅ Correct endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise if status != 200

    data = response.json()
    return [{"generated_text": data["choices"][0]["message"]["content"]}]
