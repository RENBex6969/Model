from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    response_ids = model.generate(inputs['input_ids'], max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(response_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

@app.route('/', methods=['GET'])
def get_response():
    prompt = request.args.get('prompt')
    if prompt:
        response = generate_response(prompt)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No prompt provided'})

if __name__ == '__main__':
    app.run(debug=True)
