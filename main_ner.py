import nltk
import spacy
import string
from pyvi import ViTokenizer
# Load mô hình đã huấn luyện
nlp = spacy.load("brand_price_ner_model")



def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(word.strip() for word in file.readlines())
    return stopwords

def remove_stopwords(sentence, stopwords):

    # Chuẩn hóa: loại bỏ dấu câu, chuyển về chữ thường
    sentence_cleaned = sentence.translate(str.maketrans('', '', string.punctuation)).lower()

    # Tách từ (tokenization)
    words = nltk.word_tokenize(sentence_cleaned)

    # Loại bỏ stop words
    filtered_words = [word for word in words if word not in stopwords]

    # Ghép lại thành câu
    return ' '.join(filtered_words)

stopword_file = "vietnamese-stopwords.txt"

stopwords = load_stopwords(stopword_file)

while(1):
    text = input("")
    clean_text = remove_stopwords(text, stopwords)
    doc = nlp(clean_text)
    print(f"Text: {clean_text}")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")