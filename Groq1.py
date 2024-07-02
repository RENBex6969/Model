from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.io/completion"

@app.route('/query', methods=['GET'])
def query_groq():
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Missing prompt parameter"}), 400

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192"
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return jsonify({
            "request_data": data,
            "response_data": response.json()
        })
    else:
        return jsonify({
            "error": "Failed to get response from Groq API",
            "status_code": response.status_code,
            "response_data": response.json()
        }), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
