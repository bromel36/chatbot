import spacy
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")  
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Mapping dictionaries
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


def normalize_text(text):
    """Chuyển văn bản về dạng chữ thường."""
    return text.lower()

def extract_brand(text, doc):
    """
    Nhận diện thương hiệu từ văn bản bằng spaCy và kiểm tra qua từ điển.
    """
    brands = []
    # Duyệt qua các thực thể do spaCy nhận diện
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Kiểm tra thực thể là tổ chức
            candidate = ent.text.lower()
            for brand, synonyms in brand_synonyms.items():
                if candidate in synonyms:  # Đối chiếu với từ điển
                    brands.append(brand)

        # Nếu không tìm thấy qua thực thể, kiểm tra thủ công qua từ điển
    for brand, synonyms in brand_synonyms.items():
        if any(synonym in text for synonym in synonyms):
            brands.append(brand)
    return brands


# def extract_price(text):
#     """
#     Nhận diện giá tiền từ văn bản và trả về một giá trị duy nhất.
#     """

#     price_pattern = r"\b\d{1,3}([.,]\d{3})*(\s*(triệu|tr|vnđ|vnd|đồng|đ|m|millions)?)\b|\b\d+\s*(triệu|tr|vnđ|vnd|đồng|đ|m|million)\b"


#     # Sử dụng regex để tìm tất cả các số có thể là giá
#     matches = re.finditer(price_pattern, text.lower())

#     # Nếu có ít nhất một kết quả khớp, lấy kết quả đầu tiên
#     for match in matches:
#         price_str = match.group(0)
#         # Xử lý giá trị nếu chứa đơn vị tiền tệ
#         if any(unit in price_str for unit in ["triệu", "tr", "vnđ", "vnd", "đồng", "đ", "m", "millions"]):
#             price_str = re.sub(r"[^\d,]", "", price_str)
#             price_value = float(price_str.replace(",", ".")) * 1_000_000  # Quy đổi sang đồng
#         else:
#             price_str = re.sub(r"[^\d]", "", price_str)
#             if(len(price_str) < 7):
#                 return None
#             price_value = int(price_str)

#         return price_value  # Trả về giá trị duy nhất

#     return None  # Nếu không tìm thấy giá trị nào


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



def extract_price_brand(text):
    text = normalize_text(text)
    doc = nlp(text)
    specs = {}

    # Nhận diện tất cả thương hiệu
    brands = extract_brand(text, doc)
    if brands:
        specs["brands"] = brands

    # Nhận diện tất cả giá
    prices = extract_price(text)
    if prices:
        specs["prices"] = prices

    return specs


# Example usage
# while(1):
#     text = input("")
#     print(extract_price_brand(text))




