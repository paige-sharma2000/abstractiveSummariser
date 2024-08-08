import sys
import json
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Preprocessing
contraction_mapping = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
    # Add more contractions as needed
}

stop_words = set(stopwords.words('english'))

def text_cleaner(text, remove_stopwords):
    text = text.lower()
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub('"', '', text)
    text = ' '.join([contraction_mapping.get(word, word) for word in text.split()])
    text = re.sub(r"'s\b", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub('[m]{2,}', 'mm', text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return " ".join(tokens).strip()

def load_data(file_path):
    data = pd.read_csv(file_path, nrows=100000)
    data.drop_duplicates(subset=['Reviews'], inplace=True)
    data.dropna(axis=0, inplace=True)
    data['cleaned_text'] = data['Reviews'].apply(lambda x: text_cleaner(x, True))
    data['cleaned_summary'] = data['Summary'].apply(lambda x: text_cleaner(x, False))
    data = data[['cleaned_text', 'cleaned_summary']]
    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0, inplace=True)
    return data

def prepare_data(data, max_text_len, max_summary_len):
    cleaned_text = np.array(data['cleaned_text'])
    cleaned_summary = np.array(data['cleaned_summary'])

    short_text = []
    short_summary = []

    for i in range(len(cleaned_text)):
        if len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len:
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])

    df = pd.DataFrame({'text': short_text, 'summary': short_summary})
    df['summary'] = df['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

    x_train, x_val, y_train, y_val = train_test_split(
        np.array(df['text']),
        np.array(df['summary']),
        test_size=0.1,
        random_state=0,
        shuffle=True
    )
    
    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(x_train)
    x_train_seq = x_tokenizer.texts_to_sequences(x_train)
    x_val_seq = x_tokenizer.texts_to_sequences(x_val)
    x_train = pad_sequences(x_train_seq, maxlen=max_text_len, padding='post')
    x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')
    x_voc = len(x_tokenizer.word_index) + 1

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(y_train)
    y_train_seq = y_tokenizer.texts_to_sequences(y_train)
    y_val_seq = y_tokenizer.texts_to_sequences(y_val)
    y_train = pad_sequences(y_train_seq, maxlen=max_summary_len, padding='post')
    y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')
    y_voc = len(y_tokenizer.word_index) + 1

    return (x_train, x_val, y_train, y_val, x_voc, y_voc, x_tokenizer, y_tokenizer)

def build_model(latent_dim, embedding_dim, x_voc, y_voc, max_text_len, max_summary_len):
    # Encoder
    encoder_inputs = Input(shape=(max_text_len,))
    enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Attention Mechanism
    attention = Attention()
    context_vector, attention_weights = attention([encoder_outputs, decoder_outputs], return_attention_scores=True)
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, context_vector])
    
    # Output Layer
    decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    history = model.fit(
        [x_train, y_train[:, :-1]], 
        np.expand_dims(y_train[:, 1:], -1),
        epochs=50,
        callbacks=[es],
        batch_size=128,
        validation_data=([x_val, y_val[:, :-1]], np.expand_dims(y_val[:, 1:], -1))
    )
    return history

def summarize(text, model, x_tokenizer, y_tokenizer, max_text_len, max_summary_len):
    sequence = x_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_text_len, padding='post')
    pred_summary = model.predict([padded_sequence, np.zeros((1, max_summary_len))])
    pred_summary = np.argmax(pred_summary, axis=-1)
    summary = ' '.join([y_tokenizer.index_word.get(i, '') for i in pred_summary[0] if i > 0])
    return summary.replace('eostok', '').strip()

if __name__ == "__main__":
    # Load and prepare data
    data = load_data("summarised_reviews.csv")
    max_text_len = 30
    max_summary_len = 8
    x_train, x_val, y_train, y_val, x_voc, y_voc, x_tokenizer, y_tokenizer = prepare_data(data, max_text_len, max_summary_len)
    
    # Build and train model
    latent_dim = 300
    embedding_dim = 100
    model = build_model(latent_dim, embedding_dim, x_voc, y_voc, max_text_len, max_summary_len)
    train_model(model, x_train, y_train, x_val, y_val)
    
    # Process random text from stdin
    input_text = sys.stdin.read().strip()
    
    # Generate summary
    summary = summarize(input_text, model, x_tokenizer, y_tokenizer, max_text_len, max_summary_len)
    
    # Output the result
    sys.stdout.write(summary)
