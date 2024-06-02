import os

from wow import save_to_dataset_with_watermark
from utils import count_images_in_folder, gen_rnd_text


def split_image(img, fragment_width=256, fragment_height=256, saving_folder='image_fragments'):
    os.makedirs(saving_folder, exist_ok=True)
    try:
        img_width, img_height = img.size

        # Рассчитываем количество фрагментов по горизонтали и вертикали
        num_fragments_x = img_width // fragment_width
        num_fragments_y = img_height // fragment_height

        fragment_number = count_images_in_folder(saving_folder) + 1

        for i in range(num_fragments_y):
            for j in range(num_fragments_x):
                left = j * fragment_width
                upper = i * fragment_height
                right = left + fragment_width
                lower = upper + fragment_height

                # Обрезаем фрагмент
                fragment = img.crop((left, upper, right, lower))

                # Генерируем имя файла для фрагмента
                fragment_filename = f"fragment_{fragment_number}.jpg"
                fragment_path = os.path.join(saving_folder, fragment_filename)

                # Сохраняем фрагмент
                fragment.save(fragment_path)
                save_to_dataset_with_watermark(fragment_filename, fragment, gen_rnd_text(), 100)
                print(f"Fragment saved: {fragment_path}")

                fragment_number += 1

    except Exception as e:
        print(f"Failed to split image: {e}")