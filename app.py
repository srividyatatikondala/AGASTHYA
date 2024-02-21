rom flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
app = Flask(_name_)
model = load_model('C:/Users/sree/Downloads/seq2seqmodel.html')  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["Tell me about hydrogen vehicles.", "Hydrogen vehicles use hydrogen gas as a fuel source."])
max_len = model.input_shape[1]
def generate_response(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post')
    predicted_sequence = model.predict(padded_input_sequence, verbose=0)
    predicted_sequence = np.argmax(predicted_sequence, axis=-1)
    return ' '.join(tokenizer.index_word[i] for i in predicted_sequence[0] if i != 0)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return jsonify({'response': response})
if _name_ == '_main_':
    app.run(debug=True)