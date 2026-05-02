import pytesseract
from PIL import Image

MAX_INPUTS = 20


# =========================
# OCR FUNCTION
# =========================
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"❌ OCR Error for {image_path}: {e}")
        return ""


# =========================
# MAIN PROCESSOR
# =========================
def prepare_inputs_for_core(inputs):
    """
    inputs = {
        "texts": ["..."],
        "images": ["img1.png", "img2.jpg"]
    }

    OUTPUT → list of email-like dicts
    """

    emails = []

    # 📝 TEXT INPUTS
    for text in inputs.get("texts", [])[:MAX_INPUTS]:
        if text.strip():
            emails.append({
                "sender": "unknown@edu.com",  # placeholder
                "subject": "",
                "body": text
            })

    # 🖼️ IMAGE INPUTS (OCR)
    for img_path in inputs.get("images", [])[:MAX_INPUTS]:
        extracted_text = extract_text_from_image(img_path)

        if extracted_text:
            emails.append({
                "sender": "unknown@edu.com",  # placeholder
                "subject": "",
                "body": extracted_text
            })

    return emails[:MAX_INPUTS]


# =========================
# TEST
# =========================
if __name__ == "__main__":

    test_inputs = {
        "texts": [
            "Scholarship for CS students. Apply before deadline.",
            "Internship opportunity for AI students."
        ],
        "images": [
            # Example:
            # "email1.png",
            # "email2.jpg"
        ]
    }

    emails = prepare_inputs_for_core(test_inputs)

    print("\n=== OUTPUT TO CORE.PY ===\n")

    for i, email in enumerate(emails, 1):
        print(f"\n--- Email {i} ---")
        print("BODY:\n", email["body"])