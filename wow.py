import os.path

import numpy as np
from PIL import ImageDraw, ImageFont
import pywt


def compute_detail_level(block):
    """
    Вычисление меры детализации блока с помощью вейвлет-преобразования.
    """
    block_array = np.array(block)
    coeffs = pywt.dwt2(block_array, 'haar')
    LL, (LH, HL, HH) = coeffs
    return np.sum(np.abs(HH))


def embed_watermark_text(image_name, image, watermark_text, block_size, out_folder='image_fragments_watermark'):
    """
    Внедрение текстового водяного знака в блок.
    """
    font_path = "arial.ttf"
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)

    # Размер изображения
    image_width, image_height = image.size

    # Вычисление количества блоков в строке и столбце
    blocks_per_row = image_width // block_size
    blocks_per_column = image_height // block_size

    # Выбор блока с наибольшей детализацией
    detail_levels = []
    for y in range(0, image_height, block_size):
        for x in range(0, image_width, block_size):
            block = image.crop((x, y, x + block_size, y + block_size))
            detail_levels.append((compute_detail_level(block), (x, y)))

    max_detail_block = max(detail_levels, key=lambda x: x[0])[1]

    # Внедрение водяного знака
    draw.text((max_detail_block[0], max_detail_block[1]), watermark_text, fill=(255, 255, 255), font=font)

    return image


def save_to_dataset_with_watermark(image_name, image, watermark_text, block_size,
                                   out_folder):
    embed_watermark_text(image_name, image, watermark_text, block_size, out_folder).save(
        os.path.join(out_folder, image_name))
