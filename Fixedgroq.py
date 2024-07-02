from flask import Flask, request, jsonify
import requests
import random

app = Flask(__name__)

@app.route('/chat_completion', methods=['GET'])
def chat_completion():
    prompt = request.args.get('prompt', default='', type=str)
    model = "llama3-8b-8192"  # Replace with your desired model

    # Construct the URL for Groq API
    api_url = f"https://api.groq.com/openai/v1/chat/completions?prompt={prompt}&model={model}"

    # Set headers
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    try:
        # Make GET request to Groq API
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses

        if response.status_code == 200:
            return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Request failed: {e}'}), 500

    return jsonify({'error': 'Failed to fetch response from AI service'}), 500

if __name__ == '__main__':
    # Generate a random port between 5000 and 8080
    port = random.randint(5000, 8080)
    app.run(port=port, debug=True)
