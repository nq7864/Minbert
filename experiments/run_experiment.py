import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data_dir, processed_data_dir):
    # Kiểm tra xem thư mục dữ liệu thô có tồn tại không
    if not os.path.exists(raw_data_dir):
        print(f"Thư mục dữ liệu thô {raw_data_dir} không tồn tại!")
        return

    # Đọc dữ liệu thô
    raw_file_path = os.path.join(raw_data_dir, 'imdb_reviews.csv')
    if not os.path.exists(raw_file_path):
        print(f"Tệp dữ liệu thô {raw_file_path} không tồn tại!")
        return
    
    data = pd.read_csv(raw_file_path)

    # Tiền xử lý dữ liệu: loại bỏ ký tự đặc biệt và chuyển thành chữ thường
    data['review'] = data['review'].str.replace(r'[^\w\s]', '', regex=True).str.lower()

    # Chia dữ liệu thành tập huấn luyện và kiểm tra (80-20)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Kiểm tra và tạo thư mục đã xử lý nếu chưa tồn tại
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    # Lưu dữ liệu đã tiền xử lý vào thư mục processed_data
    train_data.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)

    print("Dữ liệu đã được tiền xử lý và lưu vào thư mục:", processed_data_dir)

