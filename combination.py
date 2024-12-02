import nltk
import spacy
import string
import re
import os
# Load mô hình đã huấn luyện
nlp = spacy.load("brand_price_ner_model")

brand_synonyms = {
    "asus": ["asus", "asuz", "assus", "asus rog", "asus zenbook", "asus vivobook", "asus tuf"],
    "alurin": ["alurin", "alurine", "aurin"],
    "msi": ["msi", "m.s.i", "msi gaming", "msi stealth", "msi prestige"],
    "hp": ["hp", "h.p", "h p", "hewlett packard", "hp spectre", "hp envy", "hp pavilion"],
    "lenovo": ["lenovo", "lenova", "lenovo thinkpad", "lenovo legion", "lenovo ideapad"],
    "medion": ["medion", "median"],
    "acer": ["acer", "acer predator", "accer", "acer aspire"],
    "apple": ["apple", "macbook", "mac book", "macbook pro", "macbook air", "imac", "apple mac"],
    "razer": ["razer", "raizer", "razr", "razer blade"],
    "gigabyte": ["gigabyte", "gigabit", "gigabite", "aorus", "gigabyte aero"],
    "dell": ["dell", "del", "dell xps", "alienware", "dell inspiron", "dell latitude"],
    "lg": ["lg", "l.g", "lg gram", "life's good"],
    "samsung": ["samsung", "samsung galaxy", "samsung notebook"],
    "pccom": ["pccom", "pc com", "p.c com"],
    "microsoft": ["microsoft", "surface", "microsoft surface"],
    "primux": ["primux", "primix"],
    "prixton": ["prixton", "prixtone", "priston"],
    "dynabook toshiba": ["dynabook toshiba", "dynabook", "toshiba", "tosiba", "dynabook tosh"],
    "thomson": ["thomson", "thomsan"],
    "denver": ["denver", "denwer"],
    "deep gaming": ["deep gaming", "deep game", "deepgame"],
    "vant": ["vant", "vaant"],
    "innjoo": ["innjoo", "injoo", "inju"],
    "jetwing": ["jetwing", "jet wing"],
    "millenium": ["millenium", "milenium", "millennium"],
    "realme": ["realme", "realmi", "real me"],
}

price_pattern = r"\b\d{1,3}([.,]\d{3})*(\s*(triệu|tr|vnđ|vnd|đồng|đ|m|millions)?)\b|\b\d+\s*(triệu|tr|vnđ|vnd|đồng|đ|m|million)\b"

with open(os.path.abspath('vietnamese-stopwords.txt'), 'r', encoding='utf-8') as file:
    stop_words = set(word.strip() for word in file.readlines())
# Định nghĩa hàm extract_price
def extract_price(text):
    """
    Nhận diện giá tiền từ văn bản và trả về một giá trị duy nhất.
    """
    prices = []

    # Sử dụng regex để tìm tất cả các số có thể là giá
    matches = re.finditer(price_pattern, text.lower())

    # Nếu có ít nhất một kết quả khớp, lấy kết quả đầu tiên
    for match in matches:
        price_str = match.group(0)
        # Xử lý giá trị nếu chứa đơn vị tiền tệ
        if any(unit in price_str for unit in ["triệu", "tr", "vnđ", "vnd", "đồng", "đ", "m", "millions"]):
            price_str = re.sub(r"[^\d,]", "", price_str)
            price_value = float(price_str.replace(",", ".")) * 1_000_000  # Quy đổi sang đồng
            prices.append(price_value)
        else:
            price_str = re.sub(r"[^\d]", "", price_str)
            if(len(price_str) >= 7):
                price_value = int(price_str)
                prices.append(price_value)

    return prices  # Nếu không tìm thấy giá trị nào


def extract_brand(text):
    """
    Nhận diện thương hiệu từ văn bản.
    """
    brands = []
    # Duyệt qua các từ để nhận diện thương hiệu
    for brand, synonyms in brand_synonyms.items():
        if any(synonym in text.lower() for synonym in synonyms):
            brands.append(brand)
    return [] if not brands else brands


# Hàm xử lý chính
def process_input(text):
    """
    Xử lý input bằng cách kết hợp mô hình NER và hậu xử lý.
    """

    text = remove_stopwords(text, stop_words)

    doc = nlp(text)
    ner_prices = []
    ner_brands = []

    for ent in doc.ents:
        if ent.label_ == "PRICE":
            ner_prices.append(ent.text)
        elif ent.label_ == "BRAND":
            ner_brands.append(ent.text)

    print("brand ner: " ,ner_brands)
    print("price ner: " ,ner_prices)


    processed_prices = []
    for price in ner_prices:
        processed_prices.extend(extract_price(price))

    processed_brands = []
    for brand in ner_brands:
        processed_brands.extend(extract_brand(brand))

    return {
        "prices": list(filter(None, processed_prices)),
        "brands": list(filter(None, processed_brands))
    }

# Load stopwords
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(word.strip() for word in file.readlines())
    return stopwords


# Hàm loại bỏ stopwords
def remove_stopwords(sentence, stopwords):
    sentence_cleaned = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
    words = nltk.word_tokenize(sentence_cleaned)
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)


# Main
# if __name__ == "__main__":
#     while True:
#         text = input("Nhập câu: ")
#         result = process_input(text)
#         print(result)
