"""
CPSC 5990 Assignment 3
Long Short-Term Memory Network for text classification using TensorFlow(Keras)
By David Adams
February 29, 2020
"""

# import general packages
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

# import nltk packages
import nltk
import random
from nltk.corpus import brown, stopwords
import string
from nltk.stem import WordNetLemmatizer

# import keras packages
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# make sure corpus and stopwords collection are downloaded
nltk.download('brown')
nltk.download('stopwords')
# ---------------------------------------------------------------------------------


def plot_history(history_data):
    accuracy = history_data.history['accuracy']
    val_accuracy = history_data.history['val_accuracy']
    loss = history_data.history['loss']
    val_loss = history_data.history['val_loss']
    x = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'b', label='Training accuracy')
    plt.plot(x, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('RNN_accuracy_loss.png')
    plt.show()


# convert data into list of tuples containing a sentence (list) and a genre (string)
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
dataset = []
for genre in brown.categories():
    for sent in brown.sents(categories=genre):
        sentstr = ' '.join([lemmatizer.lemmatize(word.lower()) for word in sent if word not in stopwords
                            and word not in string.punctuation and
                            len(lemmatizer.lemmatize(word)) >= 3])
        dataset.append((sentstr, genre))

random.shuffle(dataset)


# convert dataset list to dataframe
ds = pd.DataFrame(list(dataset), columns=["sentence", "genre"])


# define input and output for model training
sentences = ds['sentence'].values
genres = ds['genre'].values


# split the training and test data using sklearn utility
sentences_train, sentences_test, genres_train, genres_test = train_test_split(sentences, genres, test_size=0.2)
# split the test data into halves, for testing and validation
sentences_test, sentences_val, genres_test, genres_val = train_test_split(sentences_test, genres_test, test_size=0.5)


# embed the sentence words into equal-sized vectors for processing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_train)
maxpad = 50
X_train = pad_sequences(tokenizer.texts_to_sequences(sentences_train), maxlen=maxpad)
X_test = pad_sequences(tokenizer.texts_to_sequences(sentences_test), maxlen=maxpad)
X_val = pad_sequences(tokenizer.texts_to_sequences(sentences_val), maxlen=maxpad)
word_index = tokenizer.word_index

vocab_size = len(word_index) + 1


# now one-hot encode the categories so they work with the model
encoder = LabelEncoder()
encoder.fit(genres_train)
label_train = encoder.transform(genres_train)
label_test = encoder.transform(genres_test)
label_val = encoder.transform(genres_val)
Y_train = to_categorical(label_train, num_classes=15)
Y_test = to_categorical(label_test, num_classes=15)
Y_val = to_categorical(label_val, num_classes=15)


# use pre-trained word2vec embedding for weights
word_vectors = KeyedVectors.load_word2vec_format \
    ('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)
EMBEDDING_DIM = 300
vocabulary_size = min(len(word_index) + 1, vocab_size)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= vocab_size:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)
del word_vectors


# Define model
model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length=maxpad,
                           trainable=True))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(15, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Now train the model
history = model.fit(X_train, Y_train,
                    epochs=2,
                    verbose=False,
                    validation_data=(X_val, Y_val),
                    batch_size=50)

# Now evaluate the model
loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
