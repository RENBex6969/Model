from flask import Flask, request, jsonify
from transformers import LlamaForConditionalGeneration, LlamaTokenizer
import random

app = Flask(__name__)

# Load LLaMA 80B model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("allenai/llama-80")
model = LlamaForConditionalGeneration.from_pretrained("allenai/llama-80")

def generate_response(prompt):
    # Limiting prompt length to avoid tokenization issues
    if len(prompt) > 1024:
        prompt = prompt[:1024]
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/', methods=['GET'])
def get_response():
    prompt = request.args.get('prompt')
    if prompt:
        response = generate_response(prompt)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No prompt provided'})

if __name__ == '__main__':
    # Generate a random port within the range 5000-8080
    port = random.randint(5000, 8080)
    app.run(debug=True, port=port)
