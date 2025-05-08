import json
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# Load dataset
with open('own.json', 'r') as f:
    data = json.load(f)

questions = [item['question'].lower() for item in data]
answers = ['<start> ' + item['answer'].lower() + ' <end>' for item in data]

# Tokenizer
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Tokenize and pad
encoder_input_data = tokenizer.texts_to_sequences(questions)
decoder_input_data = tokenizer.texts_to_sequences(answers)
decoder_target_data = [seq[1:] for seq in decoder_input_data]

max_encoder_len = max([len(seq) for seq in encoder_input_data])
max_decoder_len = max([len(seq) for seq in decoder_input_data])

encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_len, padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_decoder_len, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_decoder_len, padding='post')

decoder_target_one_hot = np.zeros((len(decoder_target_data), max_decoder_len, VOCAB_SIZE), dtype='float32')
for i, seq in enumerate(decoder_target_data):
    for t, word_idx in enumerate(seq):
        if word_idx > 0:
            decoder_target_one_hot[i, t, word_idx] = 1.0

# Split dataset
enc_train, enc_val, dec_in_train, dec_in_val, dec_out_train, dec_out_val = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_one_hot, test_size=0.2)

# Build the model
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(VOCAB_SIZE, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(VOCAB_SIZE, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit([enc_train, dec_in_train], dec_out_train,
          batch_size=64,
          epochs=100,
          validation_data=([enc_val, dec_in_val], dec_out_val))

# model.save('lstm_chatbot.h5')

# Inference models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '')
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

def chatbot_response(user_input):
    seq = tokenizer.texts_to_sequences([user_input.lower()])
    padded = pad_sequences(seq, maxlen=max_encoder_len, padding='post')
    return decode_sequence(padded)

if __name__ == "__main__":
    while True:
        input_text = input("You: ")
        if input_text.lower() in ['exit', 'quit']:
            break
        print("Bot:", chatbot_response(input_text))
