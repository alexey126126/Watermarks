import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam


def latest_checkpoint_in_directory(directory):
    checkpoints = [f for f in os.listdir(directory) if f.endswith('.weights.h5')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    return os.path.join(directory, latest_checkpoint)


def run_all():
    # Функция для загрузки и предобработки изображений
    def load_images_from_dir(directory):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = load_img(os.path.join(directory, filename), target_size=(256, 256))
                img_array = img_to_array(img) / 255.0  # Нормализация значений пикселей
                images.append(img_array)
        return np.array(images)

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

    # Пути к директориям с изображениями
    train_dir_with_watermark = 'downloader/image_fragments_watermark'
    train_dir_original = 'downloader/image_fragments'

    # Загрузка изображений
    train_images_with_watermark = load_images_from_dir(train_dir_with_watermark)
    train_images_original = load_images_from_dir(train_dir_original)

    # Построение и компиляция модели
    autoencoder = autoencoder_model()
    autoencoder.compile(optimizer=Adam(), loss='mse')

    # Определение обратного вызова для сохранения модели
    checkpoint_path = "training_checkpoint/cp-{epoch:04d}.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Если есть предыдущий чекпоинт, загрузить его и продолжить обучение
    latest_checkpoint = latest_checkpoint_in_directory(checkpoint_dir)
    if latest_checkpoint is not None:
        autoencoder.load_weights(latest_checkpoint)
        print(f"Resuming training from checkpoint {latest_checkpoint}...")
        # Определение номера последней эпохи из имени файла чекпоинта
        start_epoch = int(latest_checkpoint.split('-')[-1].split('.')[0])
        remaining_epochs = 100 - start_epoch
        if remaining_epochs > 0:
            print(f"Resuming training for {remaining_epochs} remaining epochs...")
            autoencoder.fit(train_images_with_watermark, train_images_original, epochs=100, initial_epoch=start_epoch,
                            batch_size=16, validation_split=0.2, callbacks=[cp_callback])
    else:
        # Обучение модели с использованием обратного вызова ModelCheckpoint
        autoencoder.fit(train_images_with_watermark, train_images_original, epochs=1, batch_size=32,
                        validation_split=0.2, callbacks=[cp_callback])

    # Сохранение окончательной модели
    autoencoder.save('autoencoder_model.keras')
