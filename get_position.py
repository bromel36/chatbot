
import json

def tim_vi_tri_dau_tien(chuoi_chinh, chuoi_con):
    vi_tri = chuoi_chinh.find(chuoi_con)
    return vi_tri

# Ví dụ sử dụng
# chuoi_chinh = "Tôi muốn mua laptop Dell khoảng 20 triệu."
# chuoi_con = "20"
# vi_tri = tim_vi_tri_dau_tien(chuoi_chinh, chuoi_con)

# if vi_tri != -1:
#     print(vi_tri)

# Đọc dữ liệu huấn luyện
with open("updated_data.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)["data"]

for item in training_data:
    text = item["text"]
    print(text)
    str = input("")
    print(tim_vi_tri_dau_tien(text, str))
    print(text)
    str = input("")
    print(tim_vi_tri_dau_tien(text, str))


