from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
import random

app = Flask(__name__)

# Load Bart model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def generate_response(prompt):
    # Limiting prompt length to avoid tokenization issues
    if len(prompt) > 1024:
        prompt = prompt[:1024]
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate text with additional parameters for control
    outputs = model.generate(
        input_ids=input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Prevent repetition of bigrams
        early_stopping=True,     # Stop generation if model gets stuck
        do_sample=True,         # Enable sampling for diverse responses
        temperature=0.7,        # Control randomness (0.0=deterministic, 1.0=more random)
    )
    
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
