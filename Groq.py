from flask import Flask, request, jsonify
import requests

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

    # Make GET request to Groq API
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Failed to fetch response from AI service'})

if __name__ == '__main__':
    app.run(debug=True)
