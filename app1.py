from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-1 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route('/', methods=['GET'])
def generate_response():
    prompt = request.args.get('prompt')
    if prompt:
        response = generate_text(prompt)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No prompt provided'})

if __name__ == '__main__':
    app.run(debug=True)
