
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
eng_texts, ger_texts = [], []

with open("data.txt", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            eng, ger = line.strip().split("\t")
            eng_texts.append(eng.lower())
            ger_texts.append("<start> " + ger.lower() + " <end>")

# ---------------- TOKENIZATION ----------------
eng_tok = Tokenizer(oov_token="<unk>")
ger_tok = Tokenizer(filters='', oov_token="<unk>")

eng_tok.fit_on_texts(eng_texts)
ger_tok.fit_on_texts(ger_texts)

eng_seq = eng_tok.texts_to_sequences(eng_texts)
ger_seq = ger_tok.texts_to_sequences(ger_texts)

max_eng_len = max(len(x) for x in eng_seq)
max_ger_len = max(len(x) for x in ger_seq)

eng_seq = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
ger_seq = pad_sequences(ger_seq, maxlen=max_ger_len, padding='post')

eng_vocab = len(eng_tok.word_index) + 1
ger_vocab = len(ger_tok.word_index) + 1

# ---------------- MODEL ----------------
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(eng_vocab, 128)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(ger_vocab, 128)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(ger_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# ---------------- TRAIN (silent) ----------------
model.fit(
    [eng_seq, ger_seq[:, :-1]],
    ger_seq[:, 1:],
    batch_size=32,
    epochs=100,
    verbose=0   # 🔇 no epoch logs
)

# ---------------- INFERENCE ----------------

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2,
    initial_state=decoder_states_inputs
)

decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

index_to_word = {v: k for k, v in ger_tok.word_index.items()}

# ---------------- TRANSLATE ----------------
def translate(sentence):
    seq = eng_tok.texts_to_sequences([sentence.lower()])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')

    states = encoder_model.predict(seq, verbose=0)

    target_seq = np.array([[ger_tok.word_index['<start>']]])
    result = []

    for _ in range(max_ger_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states, verbose=0
        )

        sampled_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_to_word.get(sampled_index, "")

        if sampled_word == "<end>" or sampled_word == "":
            break

        result.append(sampled_word)

        target_seq = np.array([[sampled_index]])
        states = [h, c]

    return " ".join(result)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    return jsonify({"translation": translate(text)})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=False)

