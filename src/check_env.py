import importlib


def main():
    modules = [
        "chromadb",
        "sentence_transformers",
        "pymupdf4llm",
        "rank_bm25",
        "requests",
        "tqdm",
    ]

    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"{module_name}: OK")
        except Exception as e:
            print(f"{module_name}: ERROR")
            print(e)

    print("\nПроверка окружения завершена.")


if __name__ == "__main__":
    main()  