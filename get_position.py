
import json

def find_index(str, sub):
    return str.find(sub)

# Đọc dữ liệu huấn luyện
with open("updated_data.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)["data"]

for item in training_data:
    text = item["text"]
    print(text)
    str = input("")
    print(find_index(text, str))
    print(text)
    str = input("")
    print(find_index(text, str))


