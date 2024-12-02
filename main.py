import nltk

nltk.download('popular')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from flask_cors import CORS
from keras.models import load_model
import os
import json
import random
import string
import re
from static.emo_unicode import UNICODE_EMOJI
from spec import extract_price_brand
from flask import jsonify

url_pattern = re.compile(r'http\S+')
emoji_pattern = re.compile('|'.join(UNICODE_EMOJI), flags=re.UNICODE)



type = 'demand'
type_spec = 'specification'
model = load_model(os.path.abspath('model/' + type + '/model.keras'))
modelSpec = load_model(os.path.abspath('model/' + type_spec + '/model.keras'))

# intents = json.loads(open('data.json').read())
intents = json.loads(open(os.path.abspath('data/data_demand.json'), encoding="utf8").read())
words = pickle.load(open(os.path.abspath('model/' + type + '/texts.pkl'), 'rb'))
classes = pickle.load(open(os.path.abspath('model/' + type + '/labels.pkl'), 'rb'))

intents_spec = json.loads(open(os.path.abspath('data/data_specification.json'), encoding="utf8").read())
words_spec = pickle.load(open(os.path.abspath('model/' + type_spec + '/texts.pkl'), 'rb'))
classes_spec = pickle.load(open(os.path.abspath('model/' + type_spec + '/labels.pkl'), 'rb'))



with open(os.path.abspath('vietnamese-stopwords.txt'), 'r', encoding='utf-8') as file:
    stop_words = set(word.strip() for word in file.readlines())

def clean_up_sentence(sentence):
    # ignore special characters
    sentence_trans = sentence.translate(str.maketrans('', '', string.punctuation))
    # ignore link
    sentence_trans = url_pattern.sub(r'', sentence_trans)
    # ignore emotions
    sentence_trans = emoji_pattern.sub(r'', sentence_trans)


    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence_trans)

    sentence_words = [word for word in sentence_words if word.lower() not in stop_words]


    # stem each word - create short form for word
    # sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    print(sentence_words)

    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                print(w)
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model, words, classes):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.35
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    print(ints)

    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def create_result(message, tag, brand, price):
    response_json = {
        "message": '' if not message else message,
        "tag": '' if not tag else tag,
        "brand": [] if not brand else brand,
        "price": [] if not price else price,
    }
    return response_json


def chatbot_response(msg):
    ints = predict_class(msg, model, words, classes)

    ints_spec = predict_class(msg, modelSpec, words_spec, classes_spec)

    message = ""
    tag = None
    brand = None
    price = None

    if ints:
        tag = ints[0]['intent']
        message = getResponse(ints, intents)
        if tag in ['greeting', 'no-content']:
            tag = ''

    if ints_spec:
        spec_tag = ints_spec[0]['intent']
        message = getResponse(ints_spec, intents_spec)
        if spec_tag == 'find_laptop':
            specs = extract_price_brand(msg)  # Lấy dictionary trả về từ hàm
            brand = specs.get("brands", "")
            print(brand)
            price = specs.get("prices", "")
            print(price)
    else:
        message = "Tôi không hiểu câu hỏi của bạn."

    return create_result(message, tag, brand, price)


from flask import Flask, render_template, request

app = Flask(__name__)
cors = CORS(app)
app.static_folder = 'static'


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    res = chatbot_response(userText)

    return jsonify(res)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
