"""
Проверка подключения к Ollama из Python.

Этот скрипт отправляет простой запрос к локальной Ollama
и печатает ответ модели.
"""

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


def test_ollama():
    print(f"Проверка Ollama: {OLLAMA_URL}")
    print(f"Модель: {OLLAMA_MODEL}")
    print("Отправка тестового запроса...\n")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Напиши одно слово: работает",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": 128,
            "repeat_penalty": 1.05,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()

        print("=== Ответ модели ===")
        print(answer)
        print("\nOllama работает корректно.")

    except requests.exceptions.ConnectionError:
        print("Ошибка: не удалось подключиться к Ollama.")
        print("Проверь, запущен ли Ollama-сервер.")
        print("Обычно он должен быть доступен по адресу:")
        print("http://localhost:11434")

    except requests.exceptions.Timeout:
        print("Ошибка: Ollama не ответила вовремя.")
        print("Возможно, модель слишком долго загружается или генерирует ответ.")

    except Exception as e:
        print("Ошибка при обращении к Ollama:")
        print(e)


if __name__ == "__main__":
    test_ollama()