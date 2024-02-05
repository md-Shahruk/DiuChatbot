# import all necessary libraries
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
# from tensorflow.keras.optimizers import SGD
import random

# Model processing start here

lemmatizer = WordNetLemmatizer()

# Load intents from all JSON files

# for load json files and intents basically here collect json file intents field tag,patterns
intents = []
jsonFiles = ['generalQuestion.json', 'diuCofc.json', 'tuition_fee_data.json', 'varsity_locations.json',
             'international&career.json', 'admission.json', 'campus.json', 'teacher_info.json',
             'department_office_info.json']
for json_file in jsonFiles:
    data_file = open(json_file).read()
    intents.extend(json.loads(data_file)['intents'])

# for storing words, intents tag,intents patterns, ignore words
words = []
classes = []
documents = []
ignore_words = ['?', '!', '/', '#', '(', ')', '.']

for intent in intents:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
flat_classes = [item for sublist in classes for item in sublist]
unique_classes = sorted(list(set(flat_classes)))
# classes = sorted(list(set(classes)))

# Creating trainning data
# Create empty list for storing t data
training = []
# create an empty array for number of unique intent
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
training = np.array(training, dtype=object)

# Shuffle the training data
np.random.shuffle(training)

# Split the training data into features (X) and labels (Y)
train_x = np.array(training[:, 0].tolist())
train_y = np.array(training[:, 1].tolist())

print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))
#
# model = Sequential()
# model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
from tensorflow.keras.optimizers.legacy import SGD
# rom tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers.schedules import ExponentialDecay

# configuring and compiling the neural network model
# Learning rate schedule
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

sgd = SGD(learning_rate=lr_schedule, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
# save the model
# fitting and saving the mode
history = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, validation_split=0.2, verbose=1)
model.save('model.h5')
# history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1, validation_split=0.2
#                     )
# model.save('model.h5', history)
import pickle

# Save the words (train_x) to 'texts.pkl'
with open('texts.pkl', 'wb') as texts_file:
    pickle.dump(words, texts_file)

# Save the classes (train_y) to 'labels.pkl'
with open('labels.pkl', 'wb') as labels_file:
    pickle.dump(classes, labels_file)

################################### Accuracy test #######################################
# Print and visualize training history (accuracy and loss)
print(f"Model Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
