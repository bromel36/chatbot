import json
import random

# Từ điển thương hiệu và các biến thể
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
    "realme": ["realme", "realmi", "real me"]
}

# Các mẫu câu
sentence_templates = [
    "Laptop {} giá {}.",
    "Tôi muốn mua laptop {} khoảng {}.",
    "Bạn có máy {} giá {} không?",
    "Máy {} tầm giá {} liệu có tốt không?",
    "Laptop {} giá chỉ {} thôi.",
]

# Sinh giá tiền ngẫu nhiên
def generate_price():
    prices = [
        "15 triệu", "20 triệu", "25 triệu", "30 triệu",
        "40 triệu", "50 triệu", "60 triệu", "70 triệu",
        "15.000.000 VNĐ", "20.000.000 VNĐ", "25.000.000 VNĐ",
        "23tr", "33tr", "22 tr", "43 củ"
    ]
    return random.choice(prices)

# Thuật toán sinh dữ liệu
def generate_data():
    data = []
    for brand, variations in brand_synonyms.items():
        for variation in variations:
            variation_lower = variation.lower()  # Tạo biến thể viết thường
            variation_upper = variation.upper()  # Tạo biến thể viết hoa
            for template in sentence_templates:
                price = generate_price()
                text = template.format(variation, price)
                start_brand = text.find(variation)
                end_brand = start_brand + len(variation)
                start_price = text.find(price)
                end_price = start_price + len(price)
                data.append({
                    "text": text,
                    "entities": [
                        {"start": start_brand, "end": end_brand, "label": "BRAND"},
                        {"start": start_price, "end": end_price, "label": "PRICE"}
                    ]
                })
                # Thêm trường hợp với chữ thường
                text_lower = template.format(variation_lower, price)
                start_brand_lower = text_lower.find(variation_lower)
                end_brand_lower = start_brand_lower + len(variation_lower)
                start_price_lower = text_lower.find(price)
                end_price_lower = start_price_lower + len(price)
                data.append({
                    "text": text_lower,
                    "entities": [
                        {"start": start_brand_lower, "end": end_brand_lower, "label": "BRAND"},
                        {"start": start_price_lower, "end": end_price_lower, "label": "PRICE"}
                    ]})
    return data

# Ghi dữ liệu ra file JSON
def save_data_to_json(data, file_name="augmented_data.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=2)

# Sinh dữ liệu và lưu file
if __name__ == "__main__":
    generated_data = generate_data()
    save_data_to_json(generated_data)
    print(f"Generated {len(generated_data)} sentences with labeled entities.")
