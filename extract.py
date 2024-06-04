import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Функция для извлечения водяного знака
def extract_watermark(image_path, model_path):
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))  # Масштабирование изображения до размера, который ожидает модель

    # Преобразование значений пикселей в диапазон [0, 1]
    image = image.astype('float32') / 255.0

    # Добавление размерности для соответствия ожидаемому формату модели (батч размерности)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Загрузка обученной модели автоэнкодера
    model = load_model(model_path)

    # Предсказание с использованием модели
    predicted = model.predict(image)

    # Преобразование предсказанного изображения обратно в формат изображения
    predicted_image = (predicted.squeeze() * 255).astype(np.uint8)

    return predicted_image

# Путь к изображению, из которого нужно извлечь водяной знак
image_path = 'downloader/image_fragments_watermark/fragment_1.jpg'

# Путь к обученной модели автоэнкодера
model_path = 'autoencoder_model_2.h5'

# Извлечение водяного знака
extracted_watermark = extract_watermark(image_path, model_path)

# Сохранение извлеченного водяного знака
cv2.imwrite('extracted_watermark.jpg', extracted_watermark)