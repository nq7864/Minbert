import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize

# Danh sách từ dừng (stopwords) có thể tùy chỉnh
stopwords = ['và', 'của', 'là', 'theo', 'để', 'với', 'trong', 'cho', 'đã', 'này', 'một', 'cái']

def preprocess_data(raw_data_dir, processed_data_dir):
    # Đọc dữ liệu thô
    data = pd.read_csv(os.path.join(raw_data_dir, 'imdb_reviews.csv'))

    # Tiền xử lý dữ liệu (loại bỏ các ký tự đặc biệt, chuyển thành chữ thường)
    data['review'] = data['review'].apply(lambda x: clean_text(x))

    # Chia dữ liệu thành train và test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Lưu dữ liệu đã tiền xử lý
    train_data.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)

    print("Dữ liệu đã được tiền xử lý và lưu vào thư mục:", processed_data_dir)

def clean_text(text):
    # Loại bỏ các ký tự đặc biệt, số, và dấu câu không cần thiết
    text = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9\s]', '', text)  # Chỉ giữ lại các ký tự tiếng Việt và chữ số

    # Chuyển tất cả thành chữ thường
    text = text.lower()

    # Tách từ bằng thư viện underthesea
    words = word_tokenize(text)

    # Loại bỏ stopwords
    words = [word for word in words if word not in stopwords]

    # Ghép lại thành câu
    return ' '.join(words)
