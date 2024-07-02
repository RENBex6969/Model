from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Endpoint for handling GET requests with query parameters
@app.route('/', methods=['GET'])
def handle_get():
    # Get the 'prompt' parameter from the query string
    prompt = request.args.get('prompt', '')

    if prompt:
        # Example endpoint URL and headers
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        # Example data to send in the POST request
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "llama3-8b-8192"
        }

        try:
            # Make the POST request to the external API
            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()
            return jsonify(response_data), response.status_code
        except requests.exceptions.RequestException as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Prompt parameter missing"}), 400

if __name__ == '__main__':
    app.run(debug=True)
