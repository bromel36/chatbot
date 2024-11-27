import re

# Từ điển thuộc tính và các giá trị phổ biến
attributes = {
    "brand": ["Dell", "HP", "Apple", "Asus"],
    "ram": ["8GB", "16GB", "32GB"],
    "storage": ["256GB", "512GB", "1TB"],
    "usage_type": ["văn phòng", "gaming"]
}

def detect_attributes(sentence):
    detected_attrs = {}

    # Kiểm tra từng thuộc tính
    for attr, values in attributes.items():
        for value in values:
            if value.lower() in sentence.lower():
                detected_attrs[attr] = value
                break

    # Regular Expression cho RAM: Tìm cụm từ dạng "16GB RAM" hoặc "RAM 16GB"
    ram_match = re.search(r'(\d{1,2}GB)\s+RAM|RAM\s+(\d{1,2}GB)', sentence, re.IGNORECASE)
    if ram_match:
        detected_attrs['ram'] = ram_match.group(1) or ram_match.group(2)

    # Regular Expression cho Storage: Tìm cụm từ dạng "256GB storage" hoặc "storage 256GB"
    storage_match = re.search(r'(\d{1,4}(GB|TB))\s+storage|storage\s+(\d{1,4}(GB|TB))', sentence, re.IGNORECASE)
    if storage_match:
        detected_attrs['storage'] = storage_match.group(1) or storage_match.group(3)

    return detected_attrs

# Ví dụ
sentence = "Tư vấn cho tôi laptop Dell với RAM 16GB và bộ nhớ 512GB dùng cho văn phòng"
attributes_detected = detect_attributes(sentence)
print(attributes_detected)
