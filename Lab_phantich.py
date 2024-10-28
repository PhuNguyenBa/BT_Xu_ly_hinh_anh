import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Đường dẫn tới mô hình MobileNetV2 đã được tinh chỉnh
MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

# Tải mô hình từ TensorFlow Hub
model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL, input_shape=(224, 224, 3))])

# Tải nhãn cho các lớp của ImageNet
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 
                                      'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(labels_path, "r") as f:
    class_names = f.read().splitlines()

# Đường dẫn tới thư mục ảnh
IMAGE_DIR = 'E:\lab_phan_loai_anh\archive\flowers'  # Thay thế bằng đường dẫn thư mục chứa ảnh hoa
TARGET_SIZE = (224, 224)  # Kích thước đầu vào của mô hình

# Hàm để xử lý và phân loại một ảnh
def classify_image(image_path):
    try:
        # Mở và chuẩn bị ảnh
        image = Image.open(image_path).convert('RGB')
        image = image.resize(TARGET_SIZE)
        
        # Chuẩn hóa ảnh
        img_array = np.array(image) / 255.0  # Chuẩn hóa về [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
        
        # Dự đoán với mô hình
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0], axis=-1)
        confidence = predictions[0][predicted_class]
        predicted_label = class_names[predicted_class]

        return image_path, predicted_label, confidence
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, None, None

# Hàm để phân loại ảnh trong thư mục bằng đa luồng
def classify_images_parallel(image_dir, max_workers=8):
    results = []
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sử dụng ThreadPoolExecutor để phân loại song song
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(classify_image, img_path): img_path for img_path in image_paths}
        
        # Hiển thị tiến trình xử lý với tqdm
        for future in tqdm(as_completed(future_to_image), total=len(future_to_image), desc="Processing Images"):
            img_path = future_to_image[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Processed {img_path}: Class {result[1]}, Confidence {result[2]:.2f}")
            except Exception as exc:
                print(f"Image {img_path} generated an exception: {exc}")
    
    return results

# Main execution
if __name__ == '__main__':
    # Phân loại các ảnh và lấy kết quả
    results = classify_images_parallel(IMAGE_DIR, max_workers=8)
    
    # Lưu kết quả vào file
    with open("classification_results.txt", "w") as file:
        for img_path, predicted_label, confidence in results:
            file.write(f"{img_path}: Class {predicted_label}, Confidence {confidence:.2f}\n")
