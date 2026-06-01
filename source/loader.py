def load_text(filepath: str) -> str:
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"text file not found : {filepath}") from exc

    if not text.strip():
        raise ValueError(f"text file is empty : {filepath}")
    else:
        print(f"text file loaded successfully : {filepath}")
        return text