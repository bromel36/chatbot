# import spacy
#
# # Tải mô hình spaCy
# nlp = spacy.load("en_core_web_sm")
#
# # Ví dụ câu để phân tích
# sentence = "I want a laptop with 16GB RAM and 512GB storage"
#
# # Phân tích câu
# doc = nlp(sentence)
#
# # Khởi tạo từ điển để lưu các thực thể nhận diện
# entities = {}
#
# # Đối với mỗi thực thể nhận diện được
# for ent in doc.ents:
#     if ent.label_ == "ORG":  # Nhận diện thương hiệu (brand)
#         entities["brand"] = ent.text
#     elif ent.label_ == "CARDINAL":  # Nhận diện dung lượng RAM hoặc ROM
#         if "RAM" in sentence:  # Nếu có từ 'RAM' trong câu, xác định là RAM
#             entities["ram"] = ent.text + "GB"
#         elif "storage" in sentence or "ROM" in sentence:  # Nếu có từ 'storage' hoặc 'ROM', xác định là ROM
#             entities["storage"] = ent.text + "GB"
#     elif ent.label_ == "PRODUCT":  # Nhận diện loại sử dụng (ví dụ: văn phòng)
#         entities["usage_type"] = ent.text
#
# # Kết quả trả về
# print(entities)

import spacy
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")  # Model cho tiếng Việt
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


# Regex to match prices in Vietnamese
# price_pattern = r"(\d+[\.,]?\d*\s*(triệu|tr|vnđ|vnd|đồng|đ)|\d{1,3}(\.\d{3})+( vnđ| vnd)?)"
price_pattern = r"(\d{1,3}([.,]\d{3})*(\s*(triệu|tr|vnđ|vnd|đồng|đ)?)|\d+\s*(triệu|tr|vnđ|vnd|đồng|đ))"


def normalize_text(text):
    """Chuyển văn bản về dạng chữ thường."""
    return text.lower()

def extract_brand(text, doc):
    """
    Nhận diện thương hiệu từ văn bản bằng spaCy và kiểm tra qua từ điển.
    """
    # Duyệt qua các thực thể do spaCy nhận diện
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Kiểm tra thực thể là tổ chức
            candidate = ent.text.lower()
            for brand, synonyms in brand_synonyms.items():
                if candidate in synonyms:  # Đối chiếu với từ điển
                    return brand
    # Nếu không tìm thấy qua thực thể, kiểm tra thủ công qua từ điển
    for brand, synonyms in brand_synonyms.items():
        if any(synonym in text for synonym in synonyms):
            return brand
    return None

def extract_price(text):
    """
    Sử dụng regex để tìm giá tiền trong văn bản.
    """
    match = re.search(price_pattern, text.lower())
    if match:
        price_str = match.group(0)
        # Xử lý giá trị có triệu hoặc không
        if "triệu" in price_str or "tr" in price_str:
            price_str = re.sub(r"[^\d,]", "", price_str)  # Loại bỏ chữ
            price_str = price_str.replace(",", ".")  # Chuyển ',' thành '.'
            return float(price_str) * 1_000_000  # Chuyển triệu sang đồng
        else:  # Trường hợp giá trị trực tiếp có dạng hàng nghìn hoặc triệu
            price_str = re.sub(r"[^\d]", "", price_str)  # Loại bỏ ký tự không phải số
            return int(price_str)
    return None


def extract_price_brand(text):
    """
    Kết hợp nhận diện thương hiệu và giá từ văn bản.
    """
    text = normalize_text(text)
    doc = nlp(text)
    specs = {}

    # Nhận diện thương hiệu
    brand = extract_brand(text, doc)
    if brand:
        specs["brand"] = brand

    # Nhận diện giá
    price = extract_price(text)
    if price:
        specs["price"] = price

    return specs

# # Example usage
# texts = [
#     "Tôi muốn mua một chiếc Dell XPS giá 25 triệu.",
#     "Cần tìm laptop HP Spectre khoảng 30 triệu đồng",
#     "Giá của Lenovo Thinkpad là 15,000,000 VNĐ",
#     "Tôi đang quan tâm đến Macbook Pro",
#     "tôi muốn mua laptop lenovo giá khoảng 15tr",
#     "laptop asuz giá khoảng 25 tr",
#     "laptop Acer 15 triệu",
#     "máy delll khoảng 54 tr"
# ]
#
# for text in texts:
#     print(f"Text: {text}\nExtracted: {extract_price_brand(text)}\n")

