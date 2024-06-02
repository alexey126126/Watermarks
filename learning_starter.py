import time
import random

from last_model import run_all

def train_with_retries(max_retries=10000):
    retries = 0
    while retries < max_retries:
        try:
            run_all()
            print("Обучение завершено успешно.")
            break
        except Exception as e:
            retries += 1
            print(f"Ошибка: {e}. Перезапуск обучения ({retries}/{max_retries})...")
            time.sleep(1)  # Задержка перед повторной попыткой
    else:
        print("Превышено количество попыток обучения. Обучение не удалось.")

if __name__ == "__main__":
    train_with_retries()