import re
from pathlib import Path


BASE_DIR = Path(r"D:\AI_Projects(RAG)")

INPUT_DIR = BASE_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "cleaned"


def clean_text(text: str) -> str:
    """
    Clean Whisper transcript while preserving the actual
    educational content.
    """

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common speech fillers when they appear as standalone words.
    filler_words = [
        r"\bum+\b",
        r"\bumm+\b",
        r"\bhmm+\b",
        r"\byou know\b",
    ]

    for pattern in filler_words:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Clean repeated spaces again
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def process_file(input_file: Path):
    output_file = OUTPUT_DIR / input_file.name

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Cleaned: {input_file.name}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print("No transcript files found.")
        return

    for file in txt_files:
        process_file(file)

    print("\nCleaning completed!")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()