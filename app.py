from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, Response

app = Flask(__name__)

# Load the LLaMA-70B model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/')
def index():
    prompt = request.args.get('prompt', '')
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the generated text as raw text
    return Response(generated_text, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
