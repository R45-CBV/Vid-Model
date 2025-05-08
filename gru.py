import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import json

# Load your dataset
with open('own.json') as f:
    data = json.load(f)

# Add <start> and <end> tokens to each answer
questions = [item['question'] for item in data]
answers = ['<start> ' + item['answer'] + ' <end>' for item in data]

# Tokenize
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(questions)
target_sequences = tokenizer.texts_to_sequences(answers)

# Pad sequences
max_encoder_seq_length = max(len(seq) for seq in input_sequences)
max_decoder_seq_length = max(len(seq) for seq in target_sequences)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=max_decoder_seq_length-1, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in target_sequences], maxlen=max_decoder_seq_length-1, padding='post')

# Build the model
embedding_dim = 256
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_gru, state_h = GRU(latent_dim, return_state=True)(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_gru, _ = GRU(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=state_h)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_gru)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

# Compile and train
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data],
          np.expand_dims(decoder_target_data, -1),
          batch_size=64, epochs=100, validation_split=0.2)

# ========== Inference Models ==========

# Encoder model for inference
encoder_model = Model(encoder_inputs, state_h)

# Decoder model for inference
decoder_state_input = Input(shape=(latent_dim,))
dec_emb_inf = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_gru_inf, state_h_inf = GRU(latent_dim, return_sequences=True, return_state=True)(dec_emb_inf, initial_state=decoder_state_input)
decoder_outputs_inf = decoder_dense(decoder_gru_inf)
decoder_model = Model([decoder_inputs, decoder_state_input], [decoder_outputs_inf, state_h_inf])

# ========== Response Generation ==========
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_decoder_seq_length, padding='post')

    state_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get('<start>', 1)

    decoded_sentence = ""

    for _ in range(max_decoder_seq_length):
        output_tokens, state_value = decoder_model.predict([target_seq, state_value], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        # print(f"Sampled word: {sampled_word}")  # Optional: debug

        if sampled_word in ('<end>', ''):
            break

        decoded_sentence += ' ' + sampled_word
        target_seq[0, 0] = sampled_token_index

    if decoded_sentence.strip() == "":
        return "Sorry, I don't understand that."

    return decoded_sentence.strip()

# ========== Chat with Your Bot ==========

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    response = generate_response(user_input)
    print("Bot:", response)
