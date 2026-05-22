import os
import re
import json
import base64
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime

from flask import session, redirect, request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from core import load_model, predict_email
from forward_to_chatbot import process_emails
from opportunity_advisor import analyze_opportunities


# ================= LOCAL OAUTH FIX =================
# Localhost only. Remove this when deploying online with HTTPS.
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


# ================= GOOGLE GMAIL CONFIG =================

GOOGLE_CLIENT_SECRET_FILE = "client_secret.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]

REDIRECT_URI = "http://127.0.0.1:5000/gmail/callback"


# ================= TOKEN PERSISTENCE (NEW) =================

GMAIL_TOKEN_STORE = "gmail_tokens.json"


def get_logged_in_email():
    return (session.get("user_email") or "").strip().lower()


def load_token_store():
    if not os.path.exists(GMAIL_TOKEN_STORE):
        return {}
    try:
        with open(GMAIL_TOKEN_STORE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_token_store(store):
    try:
        with open(GMAIL_TOKEN_STORE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("TOKEN STORE SAVE ERROR:", e)


def save_user_gmail_credentials(user_email: str, creds: Credentials):
    """
    Persist creds to disk so Gmail stays connected after refresh/restart.
    """
    user_email = (user_email or "").strip().lower()
    if not user_email or not creds:
        return

    store = load_token_store()
    store[user_email] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    save_token_store(store)


def load_user_gmail_credentials(user_email: str):
    """
    Load persisted creds from disk. Returns dict or None.
    """
    user_email = (user_email or "").strip().lower()
    if not user_email:
        return None

    store = load_token_store()
    creds_dict = store.get(user_email)

    if not isinstance(creds_dict, dict):
        return None

    # minimal validation
    if creds_dict.get("token_uri") and creds_dict.get("client_id") and creds_dict.get("scopes"):
        return creds_dict

    return None


def ensure_gmail_credentials_in_session():
    """
    If session creds are missing, restore them from local store.
    """
    user_email = get_logged_in_email()
    if not user_email:
        return False

    if "gmail_credentials" in session and isinstance(session.get("gmail_credentials"), dict):
        return True

    saved = load_user_gmail_credentials(user_email)
    if saved:
        session["gmail_credentials"] = saved
        return True

    return False


def refresh_and_persist_if_needed(creds: Credentials):
    """
    Refreshes expired token (if possible) and persists updated token.
    """
    if not creds:
        return creds

    try:
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request as GoogleRequest
            creds.refresh(GoogleRequest())

            # update session
            session["gmail_credentials"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }

            # persist to disk
            save_user_gmail_credentials(get_logged_in_email(), creds)

    except Exception as e:
        print("TOKEN REFRESH ERROR:", e)

    return creds


# ================= TOKEN-SAFE SETTINGS =================

MODEL_PATH = "edu_detector_model.pkl"

# Gmail API can fetch more because it does NOT use Groq tokens.
GMAIL_FETCH_LIMIT = 30

# Only this many filtered emails go to Groq.
AI_ANALYZE_LIMIT = 5

# Keep body short before Groq.
MAX_EMAIL_BODY_CHARS = 1000

# Emails with score >= 25 go to Groq.
MIN_FILTER_SCORE_FOR_AI = 25

# Fetch recent emails first. Python filter decides importance.
GMAIL_SEARCH_QUERY = "in:anywhere newer_than:90d"


# ================= BASIC HELPERS =================

def get_header(headers, header_name):
    header_name = header_name.lower()

    for header in headers:
        if header.get("name", "").lower() == header_name:
            return header.get("value", "")

    return ""


def decode_base64_urlsafe(data):
    try:
        if not data:
            return ""

        data = data.replace("-", "+").replace("_", "/")

        while len(data) % 4:
            data += "="

        decoded = base64.b64decode(data)
        return decoded.decode("utf-8", errors="ignore")

    except Exception:
        return ""


def extract_body_from_payload(payload):
    if not payload:
        return ""

    body_data = payload.get("body", {}).get("data")

    if body_data:
        return decode_base64_urlsafe(body_data)

    parts = payload.get("parts", [])

    plain_text = ""
    html_text = ""

    for part in parts:
        mime_type = part.get("mimeType", "")

        if mime_type == "text/plain":
            plain_text += decode_base64_urlsafe(
                part.get("body", {}).get("data", "")
            )

        elif mime_type == "text/html":
            html_text += decode_base64_urlsafe(
                part.get("body", {}).get("data", "")
            )

        elif part.get("parts"):
            nested_text = extract_body_from_payload(part)

            if nested_text:
                plain_text += nested_text

    return plain_text.strip() or html_text.strip()


def shorten_text(text, max_chars=MAX_EMAIL_BODY_CHARS):
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[Email shortened to reduce API token usage.]"


def is_rate_limit_error(error):
    error_message = str(error).lower()

    return (
        "429" in error_message
        or "rate_limit" in error_message
        or "rate limit" in error_message
        or "tokens per day" in error_message
        or "tpd" in error_message
    )


def safe_edu_prediction(sender):
    try:
        model = load_model(MODEL_PATH)
        return predict_email(model, sender)

    except Exception as e:
        print("EDU DETECTOR ERROR:", e)
        return {
            "email": sender,
            "domain": "",
            "prediction": "Unknown",
            "label": None,
            "confidence": 0.0
        }


# ================= URGENCY HELPERS =================

def get_today():
    return datetime.now().date()


def safe_parse_email_date(date_text):
    try:
        if not date_text:
            return None

        parsed = parsedate_to_datetime(date_text)

        if parsed:
            return parsed.date()

        return None

    except Exception:
        return None


def detect_deadline_urgency(text, email_date_text=""):
    text = (text or "").lower()
    today = get_today()

    urgency_score = 0
    urgency_reasons = []

    today_words = [
        "today",
        "tonight",
        "due today",
        "deadline today",
        "last date today",
        "submit today",
        "pay today",
        "quiz today",
        "exam today",
        "fee due today",
        "challan due today"
    ]

    tomorrow_words = [
        "tomorrow",
        "due tomorrow",
        "deadline tomorrow",
        "quiz tomorrow",
        "exam tomorrow",
        "submit tomorrow",
        "pay tomorrow",
        "assignment tomorrow",
        "presentation tomorrow"
    ]

    urgent_words = [
        "urgent",
        "asap",
        "immediately",
        "final reminder",
        "last reminder",
        "last chance",
        "closing soon",
        "deadline approaching",
        "avoid penalty",
        "late fee",
        "will be blocked",
        "account hold"
    ]

    this_week_words = [
        "this week",
        "within this week",
        "upcoming",
        "soon",
        "due soon",
        "in the next few days",
        "within 3 days",
        "within three days"
    ]

    if any(word in text for word in today_words):
        urgency_score += 45
        urgency_reasons.append("due today")

    if any(word in text for word in tomorrow_words):
        urgency_score += 40
        urgency_reasons.append("due tomorrow")

    if any(word in text for word in urgent_words):
        urgency_score += 35
        urgency_reasons.append("urgent wording")

    if any(word in text for word in this_week_words):
        urgency_score += 20
        urgency_reasons.append("due this week")

    day_names = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday"
    ]

    if any(day in text for day in day_names):
        urgency_score += 15
        urgency_reasons.append("specific day mentioned")

    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b"
    ]

    if any(re.search(pattern, text) for pattern in date_patterns):
        urgency_score += 18
        urgency_reasons.append("date mentioned")

    email_date = safe_parse_email_date(email_date_text)

    if email_date:
        age_days = (today - email_date).days

        if age_days <= 2:
            urgency_score += 10
            urgency_reasons.append("recent email")
        elif age_days <= 7:
            urgency_score += 5
            urgency_reasons.append("recent this week")

    urgency_score = max(0, min(urgency_score, 60))

    return urgency_score, urgency_reasons


def detect_academic_task_score(text):
    text = (text or "").lower()

    score = 0
    reasons = []

    academic_tasks = {
        "quiz": 30,
        "exam": 30,
        "midterm": 30,
        "mid term": 30,
        "final exam": 35,
        "assignment": 28,
        "submission": 25,
        "submit": 22,
        "deadline": 25,
        "lab task": 20,
        "project submission": 28,
        "viva": 25,
        "presentation": 22,
        "class test": 25,
        "fee challan": 35,
        "challan": 30,
        "fee deadline": 35,
        "tuition fee": 30,
        "payment deadline": 35,
        "dues": 20,
        "registration deadline": 28,
        "course registration": 25,
        "admit card": 25,
        "roll number slip": 25,
        "attendance shortage": 25,
        "makeup class": 16,
        "make-up class": 16,
        "result announced": 18,
        "grade": 12
    }

    for keyword, value in academic_tasks.items():
        if keyword in text:
            score += value
            reasons.append(keyword)

    score = max(0, min(score, 70))

    return score, reasons


def classify_priority(total_score, urgency_score):
    if urgency_score >= 40 or total_score >= 85:
        return "critical"

    if urgency_score >= 25 or total_score >= 70:
        return "high"

    if urgency_score >= 12 or total_score >= 45:
        return "medium"

    return "low"


# ================= GMAIL OAUTH =================

def connect_gmail():
    flow = Flow.from_client_secrets_file(
        GOOGLE_CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="false",
        login_hint=get_logged_in_email()
    )

    session["gmail_oauth_state"] = state

    return redirect(auth_url)


def gmail_callback_handler():
    state = session.get("gmail_oauth_state")

    flow = Flow.from_client_secrets_file(
        GOOGLE_CLIENT_SECRET_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI
    )

    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials

    session["gmail_credentials"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes
    }

    # NEW: persist to disk so it stays connected
    save_user_gmail_credentials(get_logged_in_email(), creds)

    return True


# ================= FETCH GMAIL EMAILS =================

def fetch_latest_emails(limit=GMAIL_FETCH_LIMIT):
    """
    Fetches recent emails from Gmail.
    Does NOT call Groq.
    """
    # NEW: restore creds from disk if session lost them
    ensure_gmail_credentials_in_session()

    if "gmail_credentials" not in session:
        return []

    creds = Credentials(**session["gmail_credentials"])
    creds = refresh_and_persist_if_needed(creds)

    service = build(
        "gmail",
        "v1",
        credentials=creds,
        cache_discovery=False
    )

    try:
        results = service.users().messages().list(
            userId="me",
            maxResults=limit,
            q=GMAIL_SEARCH_QUERY
        ).execute()

    except Exception as e:
        print("GMAIL FETCH ERROR:", e)
        return []

    messages = results.get("messages", [])

    email_items = []

    for index, msg in enumerate(messages, start=1):
        try:
            message = service.users().messages().get(
                userId="me",
                id=msg["id"],
                format="full"
            ).execute()

            payload = message.get("payload", {})
            headers = payload.get("headers", [])

            raw_sender = get_header(headers, "from")
            sender_name, sender_email = parseaddr(raw_sender)

            subject = get_header(headers, "subject") or "No subject"
            date_val = get_header(headers, "date")

            body = extract_body_from_payload(payload)

            if not body:
                body = message.get("snippet", "")

            email_items.append({
                "id": index,
                "sender": sender_email or raw_sender,
                "sender_name": sender_name,
                "subject": subject,
                "date": date_val,
                "body": body
            })

        except Exception as e:
            print("EMAIL READ ERROR:", e)
            continue

    return email_items


# ================= PRE-AI EMAIL FILTERING =================
# (Everything below is unchanged from your file.)

def calculate_filter_score(email_item):
    subject = (email_item.get("subject") or "").lower()
    body = (email_item.get("body") or "").lower()
    sender = (email_item.get("sender") or "").lower()
    sender_name = (email_item.get("sender_name") or "").lower()
    date_text = email_item.get("date", "")

    text = f"{subject} {body} {sender} {sender_name}"

    importance_score = 0
    matched_keywords = []

    important_keywords = {
        "scholarship": 35,
        "internship": 35,
        "fellowship": 35,
        "deadline": 30,
        "last date": 30,
        "due date": 28,
        "apply by": 30,
        "application deadline": 35,
        "applications open": 30,
        "applications are open": 30,
        "apply now": 25,
        "grant": 25,
        "funding": 25,
        "financial aid": 25,
        "tuition fee": 20,
        "postgraduate scholarship": 35,
        "undergraduate scholarship": 35,
        "research opportunity": 30,
        "career opportunity": 25,
        "job opportunity": 25,
        "admission": 20,
        "postgraduate": 18,
        "conference": 18,
        "call for applications": 35,
        "shortlisted": 30,
        "accepted": 30,
        "selection": 22,
        "final reminder": 35,
        "important deadline": 35,
        "submit your application": 30,
        "interview": 25,
        "confirmation needed": 25,
        "missing document": 25,
        "extension": 20
    }

    academic_score, academic_reasons = detect_academic_task_score(text)

    if academic_score:
        importance_score += academic_score
        matched_keywords.extend(academic_reasons)

    urgency_score, urgency_reasons = detect_deadline_urgency(text, date_text)

    if urgency_score:
        matched_keywords.extend(urgency_reasons)

    weak_keywords = {
        "program": 5,
        "student": 5,
        "university": 5,
        "education": 5,
        "learning": 4,
        "event": 4,
        "webinar": 4,
        "session": 4,
        "community": 3
    }

    junk_keywords = {
        "security alert": -80,
        "unusual sign-in": -80,
        "sign-in activity": -80,
        "new sign-in": -70,
        "password": -60,
        "verification code": -60,
        "quarantined messages": -80,
        "quarantine": -60,
        "library notification": -50,
        "library notifications": -50,
        "temu": -100,
        "promotion": -70,
        "promotional": -70,
        "discount": -70,
        "sale": -70,
        "newsletter": -45,
        "quarterly newsletter": -70,
        "monthly newsletter": -65,
        "unsubscribe": -40,
        "marketing": -60,
        "offer": -45,
        "deal": -45,
        "shopping": -60,
        "cart": -45,
        "order": -35,
        "receipt": -35,
        "invoice": -35,
        "coupon": -60,
        "social media": -35,
        "digest": -25
    }

    for keyword, value in important_keywords.items():
        if keyword in text:
            importance_score += value
            matched_keywords.append(keyword)

    for keyword, value in weak_keywords.items():
        if keyword in text:
            importance_score += value
            matched_keywords.append(keyword)

    junk_penalty = 0

    for keyword, value in junk_keywords.items():
        if keyword in text:
            junk_penalty += abs(value)
            matched_keywords.append(keyword)

    detector = safe_edu_prediction(sender)
    email_item["edu_detector"] = detector

    academic_sender_score = 0

    if detector.get("label") == 1:
        academic_sender_score += 18
        matched_keywords.append("edu model: educational sender")

    if ".edu" in sender or "edu." in sender:
        academic_sender_score += 12
        matched_keywords.append("edu sender")

    if (
        "university" in sender
        or "college" in sender
        or "school" in sender
        or "ac.uk" in sender
        or ".edu.pk" in sender
    ):
        academic_sender_score += 10
        matched_keywords.append("academic sender")

    clearly_promotional = any(
        word in text
        for word in [
            "temu",
            "discount",
            "sale",
            "coupon",
            "shopping",
            "unsubscribe",
            "promotional email",
            "marketing email"
        ]
    )

    if clearly_promotional:
        final_score = 0
    else:
        final_score = (
            importance_score
            + urgency_score
            + academic_sender_score
            - junk_penalty
        )

    is_newsletter = "newsletter" in text

    has_strong_signal = any(
        word in text
        for word in [
            "scholarship",
            "internship",
            "fellowship",
            "application deadline",
            "call for applications",
            "apply by",
            "grant",
            "funding",
            "financial aid",
            "quiz",
            "assignment",
            "exam",
            "fee challan",
            "deadline",
            "submission",
            "payment deadline"
        ]
    )

    if is_newsletter and not has_strong_signal:
        final_score = min(final_score, 15)

    final_score = max(0, min(final_score, 100))
    priority_label = classify_priority(final_score, urgency_score)

    email_item["importance_score"] = importance_score
    email_item["urgency_score"] = urgency_score
    email_item["academic_sender_score"] = academic_sender_score
    email_item["junk_penalty"] = junk_penalty
    email_item["priority_label"] = priority_label

    return final_score, matched_keywords


def filter_emails_before_ai(email_items, max_items=AI_ANALYZE_LIMIT):
    scored_emails = []
    filtered = []

    for email_item in email_items:
        score, matched_keywords = calculate_filter_score(email_item)

        email_item["filter_score"] = score
        email_item["matched_keywords"] = list(dict.fromkeys(matched_keywords))

        scored_emails.append(email_item)

        if score >= MIN_FILTER_SCORE_FOR_AI:
            filtered.append(email_item)

    scored_emails.sort(
        key=lambda item: (
            item.get("urgency_score", 0),
            item.get("filter_score", 0)
        ),
        reverse=True
    )

    filtered.sort(
        key=lambda item: (
            item.get("urgency_score", 0),
            item.get("filter_score", 0)
        ),
        reverse=True
    )

    selected = filtered[:max_items]

    ai_ready_items = []

    for index, item in enumerate(selected, start=1):
        ai_ready_items.append({
            "id": index,
            "sender": item.get("sender", "unknown@edu.com"),
            "sender_name": item.get("sender_name", ""),
            "subject": item.get("subject", "No subject"),
            "date": item.get("date", ""),
            "body": shorten_text(item.get("body", "")),
            "filter_score": item.get("filter_score", 0),
            "importance_score": item.get("importance_score", 0),
            "urgency_score": item.get("urgency_score", 0),
            "academic_sender_score": item.get("academic_sender_score", 0),
            "junk_penalty": item.get("junk_penalty", 0),
            "priority_label": item.get("priority_label", "low"),
            "matched_keywords": item.get("matched_keywords", []),
            "edu_detector": item.get("edu_detector", {})
        })

    return ai_ready_items, filtered, scored_emails


# ================= BASIC FALLBACK OUTPUT =================
# (unchanged)

def build_basic_ranked_output(filtered_emails):
    ranked = []

    sorted_emails = sorted(
        filtered_emails[:AI_ANALYZE_LIMIT],
        key=lambda item: (
            item.get("urgency_score", 0),
            item.get("filter_score", 0)
        ),
        reverse=True
    )

    for email_item in sorted_emails:
        score = email_item.get("filter_score", 0)
        urgency_score = email_item.get("urgency_score", 0)
        priority_label = email_item.get(
            "priority_label",
            classify_priority(score, urgency_score)
        )

        ranked.append({
            "sender": email_item.get("sender", "Unknown"),
            "subject": email_item.get("subject", "No subject"),
            "combined_result": {
                "final_score": score,
                "final_personalized_score": score,
                "personal_score": score,
                "combined": {
                    "title": email_item.get("subject", "Untitled Opportunity"),
                    "opportunity_type": detect_basic_type(email_item),
                    "summary": make_basic_summary(email_item),
                    "deadline_found": detect_basic_deadline(email_item),
                    "location": "Not mentioned",
                    "benefits": [],
                    "action_items": make_basic_actions(email_item),
                    "student_fit_reason": [
                        "Selected by Madadgar's urgency-aware pre-AI filter."
                    ],
                    "urgency_score": urgency_score / 10,
                    "priority_label": priority_label
                }
            }
        })

    ranked.sort(
        key=lambda item: (
            item["combined_result"]["combined"].get("urgency_score", 0),
            item["combined_result"]["final_personalized_score"]
        ),
        reverse=True
    )

    return {
        "ranked_opportunities": ranked,
        "source": "urgency_aware_basic_ranking_without_groq"
    }


def build_basic_advisor_output(ranked_output):
    advisor_results = []

    for item in ranked_output.get("ranked_opportunities", []):
        combined_result = item.get("combined_result", {})
        combined = combined_result.get("combined", {})
        score = combined_result.get("final_personalized_score", 0)
        urgency_score = combined.get("urgency_score", 0)

        if urgency_score >= 4 or score >= 85:
            why = "This email is urgent and should be handled today."
        elif score >= 70:
            why = "This email strongly matches an important academic or opportunity signal."
        elif score >= 25:
            why = "This email passed the urgency-aware opportunity filter and should be reviewed."
        else:
            why = "This email has some useful signals but may be less urgent."

        advisor_results.append({
            "advisor_analysis": {
                "why_it_matters": why,
                "recommended_action": combined.get("action_items", []),
                "fit_summary": combined.get("student_fit_reason", [])
            }
        })

    return {
        "advisor_results": advisor_results,
        "source": "urgency_aware_basic_advisor_without_groq"
    }


def detect_basic_type(email_item):
    text = (
        (email_item.get("subject") or "") + " " +
        (email_item.get("body") or "")
    ).lower()

    if "fee challan" in text or "challan" in text or "tuition fee" in text or "payment deadline" in text:
        return "Fee / Admin"

    if "quiz" in text or "exam" in text or "midterm" in text or "final exam" in text:
        return "Quiz / Exam"

    if "assignment" in text or "submission" in text or "project submission" in text:
        return "Assignment"

    if "scholarship" in text or "funding" in text or "grant" in text:
        return "Scholarship"

    if "internship" in text:
        return "Internship"

    if "fellowship" in text:
        return "Fellowship"

    if "conference" in text:
        return "Conference"

    if "career" in text or "job" in text:
        return "Career"

    if "admission" in text or "postgraduate" in text:
        return "Admission"

    return "Opportunity"


def detect_basic_deadline(email_item):
    text = (
        (email_item.get("subject") or "") + " " +
        (email_item.get("body") or "")
    ).lower()

    deadline_words = [
        "deadline",
        "last date",
        "due date",
        "apply by",
        "final reminder",
        "application deadline",
        "today",
        "tonight",
        "tomorrow",
        "friday",
        "sunday",
        "submit",
        "payment deadline",
        "fee deadline",
        "challan"
    ]

    for word in deadline_words:
        if word in text:
            return "Mentioned in email"

    return "Not mentioned"


def make_basic_summary(email_item):
    body = (email_item.get("body") or "").strip()
    subject = email_item.get("subject", "No subject")

    if body:
        return body[:300] + ("..." if len(body) > 300 else "")

    return f"Email related to: {subject}"


def make_basic_actions(email_item):
    text = (
        (email_item.get("subject") or "") + " " +
        (email_item.get("body") or "")
    ).lower()

    actions = []

    if "quiz" in text:
        actions.append("Start studying for the quiz immediately.")

    if "exam" in text or "midterm" in text or "final exam" in text:
        actions.append("Review syllabus and prepare a study plan.")

    if "assignment" in text or "submission" in text or "submit" in text:
        actions.append("Complete and submit the assignment before the deadline.")

    if "fee challan" in text or "challan" in text or "fee deadline" in text or "payment deadline" in text:
        actions.append("Pay the fee challan before the deadline to avoid penalty.")

    if "apply" in text or "application" in text:
        actions.append("Review application details.")

    if "deadline" in text or "last date" in text or "apply by" in text or "due date" in text:
        actions.append("Check deadline immediately.")

    if "scholarship" in text:
        actions.append("Check eligibility and required documents.")

    if "internship" in text:
        actions.append("Prepare CV/resume before applying.")

    if "fellowship" in text:
        actions.append("Review fellowship requirements.")

    if not actions:
        actions.append("Open and review this email.")

    return actions


# ================= MAIN PIPELINE FUNCTION =================

def rank_gmail_emails():
    fetched_emails = fetch_latest_emails(GMAIL_FETCH_LIMIT)

    if not fetched_emails:
        # Distinguish between "not connected" vs "connected but empty"
        ensure_gmail_credentials_in_session()
        if "gmail_credentials" not in session:
            return {
                "success": True,
                "mode": "gmail_not_connected",
                "fetched_count": 0,
                "filtered_count": 0,
                "sent_to_ai_count": 0,
                "email_items": [],
                "ranked_output": {"ranked_opportunities": []},
                "advisor_output": {"advisor_results": []},
                "note": "Gmail is not connected for this account. Click 'Connect Gmail' once."
            }

        return {
            "success": True,
            "mode": "gmail_connected_no_emails",
            "fetched_count": 0,
            "filtered_count": 0,
            "sent_to_ai_count": 0,
            "email_items": [],
            "ranked_output": {"ranked_opportunities": []},
            "advisor_output": {"advisor_results": []},
            "note": (
                "Gmail connected successfully, but Madadgar could not find recent emails. "
                "Try checking if this Gmail account has recent messages."
            )
        }

    ai_ready_items, filtered_emails, scored_emails = filter_emails_before_ai(
        fetched_emails,
        max_items=AI_ANALYZE_LIMIT
    )

    if not ai_ready_items:
        return {
            "success": True,
            "mode": "gmail_connected_no_important_emails",
            "fetched_count": len(fetched_emails),
            "filtered_count": 0,
            "sent_to_ai_count": 0,
            "email_items": scored_emails[:10],
            "ranked_output": {"ranked_opportunities": []},
            "advisor_output": {"advisor_results": []},
            "note": (
                "Gmail connected successfully. Madadgar scanned your recent emails, "
                "but no urgent quiz, assignment, fee challan, scholarship, internship, deadline, "
                "or academic opportunity emails were found."
            )
        }

    user_email = get_logged_in_email()

    try:
        try:
            ranked_output = process_emails(
                ai_ready_items,
                user_email=user_email
            )
        except TypeError:
            ranked_output = process_emails(ai_ready_items)

        advisor_output = analyze_opportunities(ranked_output)

        return {
            "success": True,
            "mode": "gmail_fetch_filter_then_ai_pipeline",
            "fetched_count": len(fetched_emails),
            "filtered_count": len(filtered_emails),
            "sent_to_ai_count": len(ai_ready_items),
            "email_items": ai_ready_items,
            "ranked_output": ranked_output,
            "advisor_output": advisor_output,
            "note": (
                "Gmail emails were fetched, urgency-ranked locally, and only important emails "
                "were sent to the AI ranking pipeline."
            )
        }

    except Exception as e:
        if is_rate_limit_error(e):
            ranked_output = build_basic_ranked_output(filtered_emails)
            advisor_output = build_basic_advisor_output(ranked_output)

            return {
                "success": True,
                "mode": "gmail_fetch_filter_basic_fallback",
                "warning": (
                    "Groq API limit reached. Gmail emails were fetched and urgency-ranked locally, "
                    "then basic ranking was used instead of AI ranking."
                ),
                "fetched_count": len(fetched_emails),
                "filtered_count": len(filtered_emails),
                "sent_to_ai_count": 0,
                "email_items": filtered_emails[:AI_ANALYZE_LIMIT],
                "ranked_output": ranked_output,
                "advisor_output": advisor_output
            }

        return {
            "success": False,
            "mode": "gmail_fetch_filter_pipeline_failed",
            "fetched_count": len(fetched_emails),
            "filtered_count": len(filtered_emails),
            "sent_to_ai_count": len(ai_ready_items),
            "email_items": ai_ready_items,
            "ranked_output": {"ranked_opportunities": []},
            "advisor_output": {"advisor_results": []},
            "error": f"Gmail connected and filtering worked, but AI pipeline failed: {str(e)}"
        }