import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import json
import pickle
import os
import re
from static.emo_unicode import UNICODE_EMOJI

# Khởi tạo BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tiền xử lý dữ liệu
def data_process_bert(rootPath, nameFile, type, max_length=128):
    documents = []
    classes = []

    # Đọc dữ liệu từ file JSON
    data_file = open(nameFile, encoding="utf8").read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tiền xử lý dữ liệu (xóa URL và emoji)
            url_pattern = re.compile(r'http\S+')
            pattern_tmp = url_pattern.sub(r'', pattern)
            emoji_pattern = re.compile('|'.join(UNICODE_EMOJI), flags=re.UNICODE)
            pattern_tmp = emoji_pattern.sub(r'', pattern_tmp)

            # Tokenize văn bản bằng BERT tokenizer với padding và truncation
            inputs = tokenizer(pattern_tmp, padding='max_length', truncation=True, max_length=max_length,
                               return_tensors="pt")

            documents.append((inputs, intent['tag']))  # Lưu inputs và nhãn
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Mã hóa các nhãn
    classes = sorted(list(set(classes)))

    # Lưu từ điển và nhãn
    save_path = os.path.join(rootPath, 'model', type)
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(classes, open(os.path.join(save_path, 'labels.pkl'), 'wb'))

    return documents, classes


# Chuẩn bị dữ liệu cho BERT
def prepare_data_for_bert(documents, classes):
    inputs_ids = []
    attention_masks = []
    labels = []

    # Tạo các input ids và attention masks cho mỗi mẫu
    for doc in documents:
        inputs = doc[0]
        # Squeeze để loại bỏ chiều dư thừa (pt)
        inputs_ids.append(inputs['input_ids'].squeeze(0).numpy())  # Lấy input_ids
        attention_masks.append(inputs['attention_mask'].squeeze(0).numpy())  # Lấy attention_mask

        label = classes.index(doc[1])
        labels.append(label)

    # Chuyển sang tensor
    inputs_ids = torch.tensor(inputs_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    return inputs_ids, attention_masks, labels


# Xây dựng mô hình sử dụng BERT
def build_bert_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    return model, optimizer


# Huấn luyện mô hình
def train_bert_model(model, optimizer, train_dataloader):
    model.train()

    for epoch in range(4):  # 4 epochs cho ví dụ đơn giản
        for batch in train_dataloader:
            batch_inputs_ids, batch_attention_masks, batch_labels = batch

            optimizer.zero_grad()

            # Tiến hành forward pass và tính loss
            outputs = model(batch_inputs_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss

            # Backpropagation và cập nhật gradient
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} completed')


# Đánh giá mô hình
def evaluate_bert_model(model, test_dataloader):
    model.eval()  # Chuyển mô hình sang chế độ evaluation

    correct_predictions = 0
    total_predictions = 0

    for batch in test_dataloader:
        batch_inputs_ids, batch_attention_masks, batch_labels = batch

        with torch.no_grad():
            outputs = model(batch_inputs_ids, attention_mask=batch_attention_masks)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        correct_predictions += (predictions == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# Lưu và tải mô hình
def save_model(model, model_path):
    model.save_pretrained(model_path)
    print("Model saved to", model_path)


def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    print("Model loaded from", model_path)
    return model


# Main function để huấn luyện và đánh giá mô hình
def main():
    # Đường dẫn file và tên file dữ liệu
    rootPath = 'D:\\NguyenMinhTien_N21DCCN188_PTITHCM\\ImageProcessing\\trainingtest'
    nameFile = 'data_demand.json'
    type = 'true'

    # Tiền xử lý dữ liệu
    documents, classes = data_process_bert(rootPath, nameFile, type)

    # Chia dữ liệu thành input_ids, attention_masks và labels
    inputs_ids, attention_masks, labels = prepare_data_for_bert(documents, classes)

    # Tạo DataLoader cho train và test (dữ liệu test đã được chuẩn bị)
    train_dataset = TensorDataset(inputs_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Xây dựng mô hình và optimizer
    model, optimizer = build_bert_model(len(classes))

    # Huấn luyện mô hình
    train_bert_model(model, optimizer, train_dataloader)

    # Đánh giá mô hình
    evaluate_bert_model(model, train_dataloader)

    # Lưu mô hình
    model_path = './bert_model'
    save_model(model, model_path)

    # Tải mô hình (ví dụ sử dụng lại)
    model = load_model(model_path)


if __name__ == '__main__':
    main()
