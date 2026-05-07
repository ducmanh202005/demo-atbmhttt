# DLP Demo Final

## Mục tiêu
Demo hệ thống DLP rút gọn nhưng có train model thật:
- PII detection bằng regex/rule-based
- Phân loại tài liệu bằng TF-IDF + Logistic Regression
- Phân tích hành vi bằng Random Forest
- Quyết định ALLOW / WARN / QUARANTINE / BLOCK

## Cách chạy
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Cấu trúc
- `train_models.py`: train và lưu model vào `models/`
- `detector.py`: load model và chạy pipeline phân tích
- `app.py`: giao diện Streamlit
- `data/text_dataset.csv`: dữ liệu text để train mô hình phân loại tài liệu

## Vì sao bản này hợp hơn để demo
- Có file train riêng
- Có model lưu sẵn dưới dạng `.joblib`
- App không train lại mỗi lần chạy
- Dễ trình bày đúng tinh thần “đã huấn luyện mô hình trước, sau đó tích hợp vào hệ thống DLP”
"# demo-atbmhttt" 
