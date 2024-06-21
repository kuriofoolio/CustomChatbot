from flask import Flask, flask , render_template, request, jsonify
from transformers import pipeline, AutoTokenizer
import torch

model = 'google/gemma-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    'text-generation', model=model, tokenizer=tokenizer,
    max_new_tokens=556, 
    do_sample=True, 
    temperature=0.7,
    top_p=0.95,
    top_k=50

    )

messages= []
app=Flask(__name__)
@app.route('/')
def chat_page():
    return render_template('chat.html')

state ={'chat_history':''}

@app.route('/chat', methods=['POST'])
def chat():
    data= request.get_json()
    chat_message =data.get('chat')
    messages.append({'role':'user', 'content':chat_message})
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print( prompt)
    
    outputs= pipeline(
    prompt, 
    max_new_tokens=56, 
    do_sample=True, 
    temperature=0.7,
    top_p=0.95,
    top_k=50)

    assistant_response = outputs[0]['generated_text'][len(prompt):]
    messages.append({'role':'assistant', 'content':assistant_response})
    return jsonify({'chat':f'{assistant_response}'})

if __name__ == '__main__':
    app.run(debug=True)