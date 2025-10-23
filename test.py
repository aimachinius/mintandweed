from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Định nghĩa đường dẫn file trọng số
# CHỌN MỘT TRONG HAI (TÙY MÔI TRƯỜNG):
MODEL_PATH = 'best.pt' 
# HOẶC TRÊN NANO: MODEL_PATH = 'path/to/your/best.engine' 

# 2. Tải mô hình
model = YOLO(MODEL_PATH)

# 3. Định nghĩa đường dẫn NGUỒN (Source)
# **ĐIỂM QUAN TRỌNG:** YOLO sẽ xử lý tất cả ảnh/video trong thư mục này.
IMAGE_PATH = 'try' 

# 4. Chạy dự đoán (Inference)
results = model.predict(
    source=IMAGE_PATH, 
    imgsz=512, 
    conf=0.25, # Ngưỡng tin cậy
    iou=0.7,   # Ngưỡng IoU
    # THAY 'cpu' BẰNG 'cuda:0' (hoặc device=0) NẾU DÙNG GPU/JETSON NANO
    device='cuda:0', 
    save=True # THÊM: Tự động lưu kết quả vào thư mục 'runs/segment/predict'
)

# 5. Xử lý và Hiển thị Kết quả
# LƯU Ý: Nếu chạy trên Jetson Nano, bạn nên loại bỏ hoặc thay thế phần Matplotlib
# vì nó chậm và nặng. Dùng 'save=True' (ở trên) là đủ để xem kết quả.
for r in results:
    # Nếu muốn hiển thị TỪNG ảnh trong Colab/Jupyter:
    im_array = r.plot()  
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(im_rgb)
    plt.title(f"YOLOv8 Segmentation Result for: {r.path.split('/')[-1]}") # Hiển thị tên file
    plt.axis('off')
    plt.show()
    
    print(f"File: {r.path.split('/')[-1]} - Số lượng đối tượng: {len(r.boxes)}")