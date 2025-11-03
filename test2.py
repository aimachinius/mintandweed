from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os

# -------------------------- HÀM TÍNH TRỌNG TÂM VÀ VẼ ----------------------------
def calculate_centroid_and_draw(r, image_to_draw_on):
    """
    Tính toán trọng tâm (x_center, y_center) từ mask đã rescaled
    và vẽ lên ảnh image_to_draw_on (BGR format).
    Trả về list các trọng tâm và ảnh đã vẽ.
    """
    centroids = []
    
    if r.masks is None or len(r.masks) == 0:
        return centroids, image_to_draw_on

    masks_data = r.masks.data.cpu().numpy()
    orig_h, orig_w = r.orig_shape
    pred_h, pred_w = masks_data.shape[1:]

    # Hệ số scale giữa mask dự đoán và ảnh gốc
    scale_x = orig_w / pred_w
    scale_y = orig_h / pred_h

    for mask_tensor in masks_data:
        mask_resized = cv2.resize(mask_tensor, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_np = (mask_resized > 0.5).astype(np.uint8) * 255

        M = cv2.moments(mask_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            cv2.circle(image_to_draw_on, (cx, cy), 5, (0, 0, 255), -1)
        else:
            centroids.append((None, None))

    return centroids, image_to_draw_on


# -------------------------- KHỞI TẠO VÀ CHẠY INFERENCE --------------------------
# CẤU HÌNH ĐẦU VÀO
MODEL_PATH = 'best.pt' 
IMAGE_PATH = 'try' 
OUTPUT_CSV_FILE = 'yolov8_segmentation_results.csv' 

# Tải mô hình
model = YOLO(MODEL_PATH)

# LƯU Ý QUAN TRỌNG CHO VIỆC LƯU ẢNH:
# Khi save=True, YOLOv8 sẽ tự động lưu các ảnh kết quả đã được vẽ Box/Mask
# vào thư mục 'runs/segment/predict'.
# Nếu bạn muốn LƯU CẢ ĐIỂM TRỌNG TÂM VÀO CÁC ẢNH NÀY,
# chúng ta cần CAN THIỆP vào quá trình này.
# CÁCH LÀM: Chạy inference KHÔNG có save=True, sau đó tự lưu ảnh bằng cv2.imwrite()
results = model.predict(
    source=IMAGE_PATH, 
    imgsz=512, 
    conf=0.25, 
    iou=0.7,   
    device='cuda:0', 
    save=False # Đặt save=False để chúng ta tự quản lý việc lưu ảnh
)

# -------------------------- GHI DỮ LIỆU VÀO FILE CSV --------------------------
print("\n" + "=" * 60)
print(f"Ghi dữ liệu phân tích vào file: {OUTPUT_CSV_FILE}")
print("=" * 60)

# Tạo thư mục đầu ra cho ảnh nếu chưa có
output_image_dir = 'runs/segment/predict_with_centroids_4'
os.makedirs(output_image_dir, exist_ok=True)


with open(OUTPUT_CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Ghi tiêu đề (Header)
    writer.writerow([
        'file_name', 
        'object_id', 
        'class_name', 
        'confidence_score',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
        'centroid_x', 'centroid_y'
    ])

    for r in results:
        file_name = os.path.basename(r.path)
        original_image = r.orig_img.copy()

        # Tính trọng tâm và vẽ
        # center_points, im_result_with_centroids = calculate_centroid_and_draw(r, original_image)
        center_points, im_result_with_centroids = calculate_centroid_and_draw(r, r.plot())

        # Lưu ảnh
        output_image_path = os.path.join(output_image_dir, file_name)
        cv2.imwrite(output_image_path, im_result_with_centroids)
        print(f"Đã lưu ảnh kết quả có trọng tâm: {output_image_path}")

        # Ghi CSV
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                score = scores[i]
                cls_id = class_ids[i]
                class_name = model.names[cls_id] if hasattr(model, "names") else f"Class {cls_id}"
                cx, cy = center_points[i] if i < len(center_points) and center_points[i] is not None else ("N/A", "N/A")
                writer.writerow([file_name, i + 1, class_name, f"{score:.4f}",
                                x1, y1, x2, y2, cx, cy])
        else:
            writer.writerow([file_name, 0, "No Object", 0,
                            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])


print(f"\nHoàn tất phân tích. Kết quả CSV tại: {OUTPUT_CSV_FILE}")
print(f"Ảnh kết quả có trọng tâm tại thư mục: {output_image_dir}")