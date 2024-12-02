import spacy
from spacy.training.example import Example
import json

# Tải mô hình spaCy cơ bản
nlp = spacy.blank("en")

# Thêm pipeline NER
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")

# Đọc dữ liệu huấn luyện
with open("data\\ner.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)["data"]

# Gắn nhãn dữ liệu cho NER
for item in training_data:
    text = item["text"]
    for ent in item["entities"]:
        ner.add_label(ent["label"])

# Chuẩn bị dữ liệu
examples = []
for item in training_data:
    text = item["text"].lower()  # Chuẩn hóa văn bản thành chữ thường
    doc = nlp.make_doc(text)
    entities = [(ent["start"], ent["end"], ent["label"]) for ent in item["entities"]]
    examples.append(Example.from_dict(doc, {"entities": entities}))



# Huấn luyện mô hình với Early Stopping và lưu mô hình tốt nhất
optimizer = nlp.begin_training()
best_loss = float("inf")  # Khởi tạo loss tốt nhất
no_improve_epochs = 0     # Số epoch không cải thiện liên tiếp
patience = 4              # Ngưỡng kiên nhẫn
best_model_path = "brand_price_ner_model"


for epoch in range(20):  # Train trong tối đa 20 epoch
    losses = {}
    for example in examples:
        nlp.update([example], drop=0.3, losses=losses)

    current_loss = losses["ner"]
    print(f"Epoch {epoch} - Loss: {current_loss}")

    # Lưu mô hình nếu loss giảm
    if current_loss < best_loss:
        best_loss = current_loss
        no_improve_epochs = 0
        nlp.to_disk(best_model_path)
        print(f"New best model saved with loss {best_loss}")
    else:
        no_improve_epochs += 1

    # Kiểm tra điều kiện dừng sớm
    if no_improve_epochs >= patience:
        print("Early stopping triggered.")
        break

print(f"Training completed. Best model saved to '{best_model_path}' with loss {best_loss}")
