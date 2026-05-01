import re
import random
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# =========================
# SETTINGS
# =========================
CSV_PATH = "dataset/domains.csv"
MODEL_PATH = "edu_detector_model.pkl"
RANDOM_STATE = 42


# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    """Lowercase and clean text."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def extract_domain_from_email(email: str) -> str:
    """
    Extract domain from an email.
    Example: abc@lums.edu.pk -> lums.edu.pk
    """
    email = clean_text(email)
    if "@" not in email:
        return ""
    return email.split("@")[-1].strip()


def is_valid_email(email: str) -> bool:
    """Basic email validation."""
    email = clean_text(email)
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# =========================
# DATA PREPARATION
# =========================
def build_training_data(csv_path: str) -> pd.DataFrame:
    """
    Build a balanced dataset:
    label = 1 -> educational institute domains from CSV
    label = 0 -> common non-educational domains
    """
    df = pd.read_csv(csv_path)

    # Check required column
    if "domain" not in df.columns:
        raise ValueError("CSV must contain a 'domain' column.")

    # Positive samples from your CSV
    edu_domains = (
        df["domain"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    positive_df = pd.DataFrame({
        "text": edu_domains,
        "label": 1
    })

    # Negative samples (non-edu domains)
    negative_domains = [
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
        "icloud.com", "protonmail.com", "aol.com", "zoho.com", "mail.com",
        "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
        "youtube.com", "tiktok.com", "netflix.com", "amazon.com", "daraz.pk",
        "google.com", "microsoft.com", "apple.com", "samsung.com", "oracle.com",
        "ibm.com", "intel.com", "nvidia.com", "tesla.com", "openai.com",
        "shopify.com", "ebay.com", "paypal.com", "reddit.com", "discord.com",
        "whatsapp.com", "telegram.org", "dropbox.com", "github.com", "gitlab.com",
        "wikipedia.org", "bbc.com", "cnn.com", "nike.com", "adidas.com",
        "foodpanda.pk", "careem.com", "uber.com", "booking.com", "airbnb.com",
        "fiverr.com", "upwork.com", "freelancer.com", "coursera.org", "udemy.com",
        "steam.com", "spotify.com", "canva.com", "notion.so", "slack.com",
        "example.com", "test.com", "company.org", "startup.io", "business.net"
    ]

    # Add more synthetic non-edu domains for variety
    prefixes = ["mail", "portal", "app", "service", "client", "secure", "shop", "cloud", "my", "go"]
    roots = ["alpha", "beta", "nova", "prime", "spark", "vision", "metro", "global", "swift", "smart"]
    suffixes = [".com", ".net", ".org", ".io", ".co", ".biz"]

    synthetic_negatives = []
    for p in prefixes:
        for r in roots:
            for s in suffixes:
                synthetic_negatives.append(f"{p}{r}{s}")

    negative_domains.extend(synthetic_negatives)
    negative_domains = list(set([d.lower().strip() for d in negative_domains]))

    # Remove anything that accidentally overlaps with edu domains
    edu_set = set(edu_domains)
    negative_domains = [d for d in negative_domains if d not in edu_set]

    # Balance negatives to similar size or a bit more
    if len(negative_domains) > len(edu_domains) * 2:
        random.seed(RANDOM_STATE)
        negative_domains = random.sample(negative_domains, len(edu_domains) * 2)

    negative_df = pd.DataFrame({
        "text": negative_domains,
        "label": 0
    })

    full_df = pd.concat([positive_df, negative_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return full_df


# =========================
# MODEL TRAINING
# =========================
def train_model(csv_path: str):
    """
    Train a domain classifier using character n-grams.
    This works well for domain patterns like .edu.pk, ac.uk, etc.
    """
    data = build_training_data(csv_path)

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5))),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== MODEL EVALUATION ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Trained model saved to: {MODEL_PATH}")

    return model


# =========================
# PREDICTION
# =========================
def predict_email(model, email: str):
    """
    Predict whether an email is from an educational institute or not.
    """
    if not is_valid_email(email):
        return {
            "email": email,
            "domain": "",
            "prediction": "Invalid email format",
            "label": None,
            "confidence": 0.0
        }

    domain = extract_domain_from_email(email)

    if not domain:
        return {
            "email": email,
            "domain": "",
            "prediction": "Could not extract domain",
            "label": None,
            "confidence": 0.0
        }

    pred = model.predict([domain])[0]
    probs = model.predict_proba([domain])[0]
    confidence = float(max(probs))

    return {
        "email": email,
        "domain": domain,
        "prediction": "Educational Institute" if pred == 1 else "Not Educational Institute",
        "label": int(pred),
        "confidence": round(confidence, 4)
    }


def load_model(model_path: str = MODEL_PATH):
    """Load saved model."""
    return joblib.load(model_path)


# =========================
# TESTING
# =========================
if __name__ == "__main__":
    # Train and save model
    model = train_model(CSV_PATH)

    # Test samples
    test_emails = [
        "admissions@qau.edu.pk",
        "info@nust.edu.pk",
        "student@lums.edu.pk",
        "hello@gmail.com",
        "support@amazon.com",
        "abc@ox.ac.uk",
        "contact@harvard.edu",
        "user@yahoo.com",
        "office@company.org"
    ]

    print("\n=== SAMPLE PREDICTIONS ===")
    for email in test_emails:
        result = predict_email(model, email)
        print(
            f"Email: {result['email']}\n"
            f"Domain: {result['domain']}\n"
            f"Prediction: {result['prediction']}\n"
            f"Confidence: {result['confidence']}\n"
        )