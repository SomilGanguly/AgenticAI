
import requests

def analyze_architecture_with_gpt4o(img, openai_endpoint, openai_api_key):
    # Prepare the image for GPT-4o (Vision)
    if "base64" in img:
        image_data = img["base64"]
    else:
        # If you have a public URL, you can use it directly
        image_data = None
        image_url = img["url"]

    prompt = (
        "Analyze this image. List all resource icons and boundaries present in this architecture diagram. "
        "Identify each tier and its components. Return the result as a structured list."
    )

    headers = {
        "api-key": openai_api_key,
        "Content-Type": "application/json"
    }

    # For Azure OpenAI, use the /openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview endpoint
    # For OpenAI, use https://api.openai.com/v1/chat/completions

    # Example payload for Azure OpenAI GPT-4o Vision
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }

    response = requests.post(
        f"{openai_endpoint}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    result = response.json()
    # Extract the model's reply
    reply = result["choices"][0]["message"]["content"]
    return reply
