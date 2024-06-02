import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

from wow import embed_watermark_text


# Функция для построения модели автокодировщика
def autoencoder_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(2, padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(2, padding='same')(x)

    x = Conv2D(32, 3, activation='relu', padding='same')(encoded)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    decoded = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    return autoencoder


# Функция для загрузки обученной модели
def load_model():
    global autoencoder
    autoencoder = autoencoder_model()
    autoencoder.load_weights('autoencoder_model.h5')
    print("Model loaded successfully.")


# Функция для загрузки изображения
def load_image():
    global img_with_watermark, img_array, original_size
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        img_with_watermark = Image.open(file_path)
        original_size = img_with_watermark.size
        img_array = np.array(img_with_watermark) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем измерение пакета
        img_display = ImageTk.PhotoImage(img_with_watermark)
        panelA.config(image=img_display)
        panelA.image = img_display


# Функция для извлечения водяного знака из блоков изображения
def extract_watermark():
    global img_result
    if img_with_watermark is not None:
        img_array_padded, pad_h, pad_w = pad_image(img_array[0], 256)
        result = process_blocks(img_array_padded, 256)
        result = result[0:original_size[1], 0:original_size[0], :]  # Убираем отступы
        img_result = Image.fromarray((result * 255).astype(np.uint8))
        img_display = ImageTk.PhotoImage(img_result)
        panelB.config(image=img_display)
        panelB.image = img_display
        messagebox.showinfo("Success", "Watermark extracted successfully!")


# Функция для добавления отступов к изображению
def pad_image(image, block_size):
    h, w, _ = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return padded_image, pad_h, pad_w


# # Функция для обработки блоков изображения
def process_blocks(image, block_size):
    h, w, _ = image.shape
    result_image = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size, :]
            block = np.expand_dims(block, axis=0)
            result_block = autoencoder.predict(block)
            result_image[i:i + block_size, j:j + block_size, :] = result_block[0]
    return result_image


# Функция для встраивания водяного знака
def embed_watermark():
    global img_with_watermark, img_array_with_watermark, img_result
    watermark_text = watermark_entry.get()
    if watermark_text:
        img_with_watermark = embed_watermark_text(img_with_watermark, watermark_text, 256)
        img_array_with_watermark = np.array(img_with_watermark) / 255.0
        img_result = img_with_watermark  # Обновляем переменную img_result
        img_display = ImageTk.PhotoImage(img_with_watermark)
        panelA.config(image=img_display)
        panelA.image = img_display
        messagebox.showinfo("Success", "Watermark embedded successfully!")
    else:
        messagebox.showwarning("Input Error", "Please enter watermark text.")


# Функция для сохранения результата
def save_result():
    global img_result
    if img_result is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            img_result.save(save_path)
            messagebox.showinfo("Success", f"Image saved successfully at {save_path}")


# Инициализация окна
root = tk.Tk()
root.title("Watermark Embedder and Extractor")

# Загрузка модели
# load_model()

# Инициализация глобальных переменных
img_with_watermark = None
img_array_with_watermark = None
img_result = None
original_size = None

# Кнопки интерфейса
btn_load = tk.Button(root, text="Load Image", command=load_image)
btn_load.pack(side="top", fill="both", expand="yes", padx=10, pady=10)

watermark_label = tk.Label(root, text="Enter Watermark Text:")
watermark_label.pack(side="top", fill="both", expand="yes", padx=10, pady=5)

watermark_entry = tk.Entry(root)
watermark_entry.pack(side="top", fill="both", expand="yes", padx=10, pady=5)

btn_embed = tk.Button(root, text="Embed Watermark", command=embed_watermark)
btn_embed.pack(side="top", fill="both", expand="yes", padx=10, pady=10)

btn_extract = tk.Button(root, text="Extract Watermark", command=extract_watermark)
btn_extract.pack(side="top", fill="both", expand="yes", padx=10, pady=10)

btn_save = tk.Button(root, text="Save Result", command=save_result)
btn_save.pack(side="top", fill="both", expand="yes", padx=10, pady=10)

# Панели для отображения изображений
panelA = tk.Label(root)
panelA.pack(side="left", padx=10, pady=10)

panelB = tk.Label(root)
panelB.pack(side="right", padx=10, pady=10)

# Запуск главного цикла
root.mainloop()
