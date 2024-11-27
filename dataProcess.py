# import nltk
# from nltk.stem import WordNetLemmatizer
#
# lemmatizer = WordNetLemmatizer()
# import json
# import pickle
# import numpy as np
# import random
# import re
# from static.emo_unicode import UNICODE_EMOJI
# import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import SGD
#
#
# def data_process(rootPath, nameFile, type):
#     words = []
#     classes = []
#     documents = []
#     ignore_words = ['?', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '.', ',', '/', ':', ';', '<', '=', '>',
#                     '?',
#                     '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', '\t', '\n', "'", '_']
#
#     data_file = open(nameFile, encoding="utf8").read()
#
#     intents = json.loads(data_file)
#
#     for intent in intents['intents']:
#         for pattern in intent['patterns']:
#             # ignore link
#             url_pattern = re.compile(r'http\S+')
#             pattern_tmp = url_pattern.sub(r'', pattern)
#
#             # ignore emotions
#             emoji_pattern = re.compile('|'.join(UNICODE_EMOJI), flags=re.UNICODE)
#             pattern_tmp = emoji_pattern.sub(r'', pattern_tmp)
#
#             # tokenize each word
#             w = nltk.word_tokenize(pattern_tmp)
#             words.extend(w)
#             # add documents in the corpus
#             documents.append((w, intent['tag']))
#
#             # add to our classes list
#             if intent['tag'] not in classes:
#                 classes.append(intent['tag'])
#
#     # lemmaztize and lower each word and remove duplicates
#     words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
#     words = sorted(list(set(words)))
#     # sort classes
#     classes = sorted(list(set(classes)))
#     # documents = combination between patterns and intents
#     print(len(documents), "documents")
#     # classes = intents
#     print(len(classes), "classes", classes)
#     # words = all words, vocabulary
#     print(len(words), "unique lemmatized words", words)
#
#     save_path = os.path.join(rootPath, 'model', type)
#     os.makedirs(save_path, exist_ok=True)
#
#     pickle.dump(words, open(os.path.join(save_path, 'texts.pkl'), 'wb'))
#     pickle.dump(classes, open(os.path.join(save_path, 'labels.pkl'), 'wb'))
#
#     # pickle.dump(words, open(rootPath + '\\model\\' + type + '\\texts.pkl', 'wb'))
#     # pickle.dump(classes, open(rootPath + '\\model\\' + type + '\\labels.pkl', 'wb'))
#
#     # pickle.dump(words, open(rootPath + '\\model\\wrong\\texts.pkl', 'wb'))
#     # pickle.dump(classes, open(rootPath + '\\model\\wrong\\labels.pkl', 'wb'))
#
#     # create our training data
#     training = []
#     # create an empty array for our output
#     output_empty = [0] * len(classes)
#     cnt = 0
#     # training set, bag of words for each sentence
#     for doc in documents:
#         # initialize our bag of words
#         bag = []
#         # list of tokenized words for the pattern
#         pattern_words = doc[0]
#         # lemmatize each word - create base word, in attempt to represent related words
#         pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#         # create our bag of words array with 1, if word match found in current pattern
#         for w in words:
#             bag.append(1) if w in pattern_words else bag.append(0)
#
#         # output is a '0' for each tag and '1' for current tag (for each pattern)
#         output_row = list(output_empty)
#         output_row[classes.index(doc[1])] = 1
#
#         # cnt = cnt + 1
#         # print(cnt)
#         # print("bag: ", len(bag))
#         # print("output_row: ", len(output_row))
#
#         training.append([bag, output_row])
#     # shuffle our features and turn into np.array
#     random.shuffle(training)
#
#     for i, item in enumerate(training):
#         if not isinstance(item[0], list) or not isinstance(item[1], list):
#             print(f"Issue at index {i}: bag type {type(item[0])}, output_row type {type(item[1])}")
#
#     training = np.array(training, dtype=object)
#     # create train and test lists. X - patterns, Y - intents
#     train_x = list(training[:, 0])
#     train_y = list(training[:, 1])
#     print("Training data created")
#     return train_x, train_y
#
#
# rootPath = 'D:\\NguyenMinhTien_N21DCCN188_PTITHCM\\ImageProcessing\\trainingtest'
# type = 'true'  # true or wrong
# train_x, train_y = data_process(rootPath, 'data_demand.json', type)
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation="softmax"))
#
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# import tensorflow as tf
#
#
# def lr_schedule(epoch):
#     learning_rate = 0.01
#     if epoch > 10:
#         learning_rate *= 0.1
#     return learning_rate
#
#
# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# # fitting and saving the model
# hist = model.fit(
#     np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1
# )
# model.save(rootPath + '\\model\\' + type + '\\model.h5', hist)
#
# print("model created")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
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

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    save_path = os.path.join(rootPath, 'model', type)
    os.makedirs(save_path, exist_ok=True)

    pickle.dump(words, open(os.path.join(save_path, 'texts.pkl'), 'wb'))
    pickle.dump(classes, open(os.path.join(save_path, 'labels.pkl'), 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
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


rootPath = 'D:\\NguyenMinhTien_N21DCCN188_PTITHCM\\ImageProcessing\\trainingtest'
type = 'specification'
nameFile = 'data\\data_specification.json'
train_x, train_y = data_process(rootPath, nameFile, type)

# # Chia dữ liệu thành train và test
# train_x, test_x, train_y, test_y = train_test_split(
#     np.array(train_x), np.array(train_y), test_size=0.1, random_state=78
# )


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


# Hàm callback để điều chỉnh learning rate
def lr_schedule(epoch):
    learning_rate = 0.01
    if epoch > 10:
        learning_rate *= 0.1
    return learning_rate


sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

# # Huấn luyện mô hình
# hist = model.fit(
#     train_x, train_y,
#     epochs=100,
#     batch_size=5,
#     verbose=1,
#     validation_data=(test_x, test_y)  # Đánh giá trên tập test
# )

# Lưu mô hình
# model.save(os.path.join(rootPath, 'model', type, 'model.h5'), hist) đã lưu ở trên

# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(test_x, test_y)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

print("Model created and evaluated")




