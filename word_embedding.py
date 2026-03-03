import os

imdb_dir = '/Users/fsr/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir , 'train')

labels = []
texts = []

for label_type in ['neg' , 'pos']:
    dir_name = os.path.join(train_dir , label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name , fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import  pad_sequences
import numpy as np

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences , maxlen = maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]

glove_dir = '/Users/fsr/Downloads/glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir , 'glove.6B.100d.txt'))
for line in f :
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:] , dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words,embedding_dim))
for word , i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

from keras.models import Sequential
from keras.layers import Embedding , Flatten , Dense

model = Sequential()
model.add(Embedding(max_words , embedding_dim , input_length = maxlen))
model.add(Flatten())
model.add(Dense(32 , activation = 'relu'))
model.add(Dense(1 , activation = 'sigmoid'))
model.build(input_shape=(None, maxlen))
model.summary()


model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])
history = model.fit(x_train , y_train,
                    epochs = 10,
                    batch_size = 32,
                    validation_data = (x_val , y_val))
model.save_weights('pre_trained_glove_model.weights.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1 , len(acc)+1)

plt.plot(epochs , acc , 'bo' , label = 'Training acc')
plt.plot(epochs , val_acc , 'b' , label = 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs , loss , 'bo' , label = 'Training loss')
plt.plot(epochs , val_loss , 'b' , label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_dir = os.path.join(imdb_dir , 'test')

labels = []
texts = []

for label_type in ['neg' , 'pos']:
    dir_name = os.path.join(test_dir , label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name , fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences , maxlen = maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.weights.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

                
