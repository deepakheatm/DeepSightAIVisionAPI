from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer

def main():
    print("Initializing PaddleOCR (downloads OCR models if not cached)...")

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en"
    )

    print("PaddleOCR initialized.")

    print("Downloading sentence-transformer models...")

    # English semantic model
    st_en = SentenceTransformer("all-MiniLM-L6-v2")
    _ = st_en.encode(["test"])
    print("English sentence-transformer model ready.")

    # Multilingual model
    st_multi = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    _ = st_multi.encode(["prueba"])
    print("Multilingual sentence-transformer model ready.")

    print("Model bootstrap complete.")

if __name__ == "__main__":
    main()
