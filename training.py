from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import re
from static.emo_unicode import UNICODE_EMOJI
import os

lemmatizer = WordNetLemmatizer()


def data_process(rootPath, nameFile, type):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '.', ',', '/', ':', ';', '<', '=', '>',
                    '?', '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', '\t', '\n', "'", '_']

    data_file = open(nameFile, encoding="utf8").read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            url_pattern = re.compile(r'http\S+')
            pattern_tmp = url_pattern.sub(r'', pattern)

            emoji_pattern = re.compile('|'.join(UNICODE_EMOJI), flags=re.UNICODE)
            pattern_tmp = emoji_pattern.sub(r'', pattern_tmp)

            w = nltk.word_tokenize(pattern_tmp)
            words.extend(w)
            documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [(w.lower()) for w in words if w not in ignore_words]

    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique words", words)

    save_path = os.path.join(rootPath, 'model', type)
    os.makedirs(save_path, exist_ok=True)

    pickle.dump(words, open(os.path.join(save_path, 'texts.pkl'), 'wb'))
    pickle.dump(classes, open(os.path.join(save_path, 'labels.pkl'), 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [word.lower() for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    print("Training data created")
    return train_x, train_y


rootPath = 'D:\\intelligent-system-analysis-design\\chatbot'
type = 'specification'
nameFile = 'data\\specification.json'
train_x, train_y = data_process(rootPath, nameFile, type)


# Sử dụng stratified sampling để phân chia dữ liệu
train_x, test_x, train_y, test_y = train_test_split(
    np.array(train_x), np.array(train_y), test_size=0.2, random_state=56, stratify=np.array(train_y)
)


# Xây dựng mô hình
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))


sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Thêm callback Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(rootPath, 'model', type, 'model.keras'),
    monitor='val_loss',
    save_best_only=True
)


# Huấn luyện mô hình với Early Stopping
hist = model.fit(
    train_x, train_y,
    epochs=100,
    batch_size=5,
    verbose=1,
    validation_data=(test_x, test_y),
    callbacks=[early_stopping, checkpoint]
)


# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(test_x, test_y)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

print("Model created and evaluated")




