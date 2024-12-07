import sys
sys.path.append('d:/BTL/MinbertSentimentAnalysis/utils')

from utils.data_preprocessing import preprocess_data
from models.minbert_model import MinBERTModel
from models.trainer import Trainer
from utils.evaluation import evaluate_model

# Đường dẫn tới các thư mục chứa dữ liệu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data/raw_data')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed_data')

def main():
    # Kiểm tra xem thư mục dữ liệu có tồn tại không
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Thư mục dữ liệu thô {RAW_DATA_DIR} không tồn tại!")
        return

    # Tiền xử lý dữ liệu thô
    print("Bắt đầu tiền xử lý dữ liệu...")
    preprocess_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    print("Dữ liệu đã được tiền xử lý và lưu vào thư mục:", PROCESSED_DATA_DIR)

    # Khởi tạo mô hình MinBERT
    print("Khởi tạo mô hình MinBERT...")
    model = MinBERTModel()

    # Khởi tạo đối tượng Trainer và huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    trainer = Trainer(model)
    trainer.train(PROCESSED_DATA_DIR)
    print("Huấn luyện mô hình hoàn tất.")

    # Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    evaluate_model(model, PROCESSED_DATA_DIR)
    print("Đánh giá mô hình hoàn tất.")

if __name__ == "__main__":
    main()
