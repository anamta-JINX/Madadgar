import os

# ================= LOCAL GOOGLE OAUTH FIX =================
# Localhost only. Remove this when deploying online with HTTPS.
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from typing import List, Dict, Any

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract

from core import load_model, predict_email
from student_profile_manager import save_student_profile, get_profile_by_email
from chatbot_api import chat
from forward_to_chatbot import process_emails
from opportunity_advisor import analyze_opportunities


# ================= APP CONFIG =================

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "madadgar-secret-key-change-this"

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True

MAX_INPUTS = 20
MODEL_PATH = "edu_detector_model.pkl"
UPLOAD_FOLDER = "temp_uploads"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ================= BASIC HELPERS =================

def get_logged_in_email() -> str:
    return (session.get("user_email") or "").strip().lower()


def safe_profile_by_email(email: str):
    """
    Important:
    This should only return the logged-in user's profile.
    It should NOT fall back to the latest/old profile.
    """
    try:
        email = (email or "").strip().lower()

        if not email:
            return None

        return get_profile_by_email(email)

    except Exception as e:
        print("PROFILE LOAD ERROR:", e)
        return None


def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def extract_text_from_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print("OCR ERROR:", e)
        return ""


def normalize_text_inputs(texts: List[str]) -> List[Dict[str, Any]]:
    items = []

    for i, text in enumerate(texts[:MAX_INPUTS], start=1):
        text = (text or "").strip()

        if text:
            items.append({
                "id": i,
                "sender": "unknown@edu.com",
                "sender_name": "",
                "subject": f"Text Input {i}",
                "date": "",
                "body": text
            })

    return items


def process_uploaded_images(files) -> List[Dict[str, Any]]:
    items = []

    for idx, file in enumerate(files[:MAX_INPUTS], start=1):
        if not file or not file.filename:
            continue

        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(temp_path)
            extracted_text = extract_text_from_image(temp_path)

            if extracted_text:
                items.append({
                    "id": idx,
                    "sender": "unknown@edu.com",
                    "sender_name": "",
                    "subject": filename,
                    "date": "",
                    "body": extracted_text
                })

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return items


def safe_number(value, default=0):
    try:
        return float(value)
    except Exception:
        return default


def process_emails_safely(email_items, user_email=""):
    try:
        return process_emails(email_items, user_email=user_email)
    except TypeError:
        return process_emails(email_items)


def call_chat_safely(user_message: str, user_email: str):
    try:
        return chat(user_message, user_email)
    except TypeError:
        return chat(user_message)


# ================= DISPLAY / DASHBOARD HELPERS =================

def extract_ranked_list(ranked_output: Any) -> List[Dict[str, Any]]:
    if isinstance(ranked_output, dict):
        ranked = ranked_output.get("ranked_opportunities", [])
        return ranked if isinstance(ranked, list) else []

    return []


def extract_advisor_results(advisor_output: Any) -> List[Dict[str, Any]]:
    if isinstance(advisor_output, dict):
        advisor_results = advisor_output.get("advisor_results", [])
        return advisor_results if isinstance(advisor_results, list) else []

    return []


def get_urgency_from_score_and_text(score, combined):
    """
    This feeds dashboard.html with clean urgency labels:
    critical / high / medium / low

    dashboard.html then turns those into:
    ACT TODAY / THIS WEEK / CAN WAIT
    """
    score = safe_number(score, 0)

    title = str(combined.get("title", "") or "").lower()
    summary = str(combined.get("summary", "") or "").lower()
    deadline = str(combined.get("deadline_found", "") or "").lower()
    action_items = " ".join(combined.get("action_items", []) or []).lower()

    text = f"{title} {summary} {deadline} {action_items}"

    today_words = [
        "today",
        "tonight",
        "urgent",
        "immediately",
        "asap",
        "final reminder",
        "last date",
        "deadline today"
    ]

    this_week_words = [
        "this week",
        "tomorrow",
        "friday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "saturday",
        "sunday",
        "soon",
        "upcoming"
    ]

    if any(word in text for word in today_words):
        return "critical"

    if score >= 85:
        return "critical"

    if score >= 70:
        return "high"

    if any(word in text for word in this_week_words):
        return "medium"

    if score >= 45:
        return "medium"

    return "low"


def build_display_opportunities(ranked_output: Any, advisor_output: Any) -> List[Dict[str, Any]]:
    """
    Converts your backend ranking output into the clean structure dashboard.html needs.

    Dashboard expects:
    - title
    - sender
    - summary
    - score
    - deadline
    - urgency
    - type
    - actions
    - benefits
    - reason
    """

    display_items = []

    ranked = extract_ranked_list(ranked_output)
    advisor_results = extract_advisor_results(advisor_output)

    for index, item in enumerate(ranked):
        if not isinstance(item, dict):
            continue

        combined_result = item.get("combined_result", {}) or {}
        combined = combined_result.get("combined", {}) or {}

        advisor_analysis = {}

        if index < len(advisor_results):
            advisor_item = advisor_results[index]

            if isinstance(advisor_item, dict):
                advisor_analysis = advisor_item.get("advisor_analysis", {}) or {}

        score = combined_result.get(
            "final_personalized_score",
            combined_result.get("final_score", 0)
        )

        urgency = get_urgency_from_score_and_text(score, combined)

        reason = ""

        if isinstance(advisor_analysis, dict):
            reason = advisor_analysis.get("why_it_matters", "")

        if not reason:
            fit_reason = combined.get("student_fit_reason", [])

            if isinstance(fit_reason, list) and fit_reason:
                reason = " ".join(str(x) for x in fit_reason[:2])
            else:
                reason = combined.get("summary", "")

        display_items.append({
            "id": index + 1,
            "rank": index + 1,

            "sender": item.get("sender", "Unknown"),
            "subject": item.get("subject", combined.get("title", "Untitled")),
            "title": combined.get("title", item.get("subject", "Untitled Opportunity")),

            "score": round(safe_number(score, 0), 1),
            "base_score": round(safe_number(combined_result.get("final_score", 0)), 1),
            "personal_score": round(safe_number(combined_result.get("personal_score", 0)), 1),

            "type": combined.get("opportunity_type", "Opportunity"),
            "summary": combined.get("summary", "No summary available."),
            "deadline": combined.get("deadline_found", "Not mentioned"),
            "location": combined.get("location", "Not mentioned"),

            "benefits": combined.get("benefits", []) if isinstance(combined.get("benefits", []), list) else [],
            "actions": combined.get("action_items", []) if isinstance(combined.get("action_items", []), list) else [],
            "fit_reason": combined.get("student_fit_reason", []) if isinstance(combined.get("student_fit_reason", []), list) else [],

            "advisor_analysis": advisor_analysis,
            "reason": reason,
            "urgency": urgency
        })

    display_items.sort(
        key=lambda item: safe_number(item.get("score", 0)),
        reverse=True
    )

    return display_items


def empty_dashboard_context():
    return {
        "user_email": get_logged_in_email(),
        "opportunities": [],
        "email_items": [],
        "ranked_output": {},
        "advisor_output": {},
        "mode": "",
        "warning": "",
        "note": "",
        "error": "",
        "fetched_count": 0,
        "filtered_count": 0,
        "sent_to_ai_count": 0
    }


def render_dashboard_with_data(data):
    ranked_output = data.get("ranked_output", {})
    advisor_output = data.get("advisor_output", {})
    display_opportunities = build_display_opportunities(ranked_output, advisor_output)

    return render_template(
        "dashboard.html",
        user_email=get_logged_in_email(),

        opportunities=display_opportunities,
        email_items=data.get("email_items", []),

        ranked_output=ranked_output,
        advisor_output=advisor_output,

        mode=data.get("mode", ""),
        warning=data.get("warning", ""),
        note=data.get("note", ""),
        error=data.get("error", ""),

        fetched_count=data.get("fetched_count", 0),
        filtered_count=data.get("filtered_count", 0),
        sent_to_ai_count=data.get("sent_to_ai_count", 0)
    )


# ================= WEBSITE ROUTES =================

@app.route("/")
def root():
    return render_template("home.html", user_email=get_logged_in_email())


@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()

    if not email:
        return redirect(url_for("login_page"))

    session.clear()
    session["user_email"] = email

    return redirect(url_for("profile_page"))


@app.route("/signup", methods=["POST"])
def signup():
    email = request.form.get("email", "").strip().lower()

    if not email:
        return redirect(url_for("login_page"))

    session.clear()
    session["user_email"] = email

    return redirect(url_for("profile_page"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("root"))


@app.route("/profile")
def profile_page():
    user_email = get_logged_in_email()
    profile = safe_profile_by_email(user_email)

    return render_template(
        "profile.html",
        user_email=user_email,
        profile=profile
    )


@app.route("/input")
def input_page():
    return render_template("input.html", user_email=get_logged_in_email())


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", **empty_dashboard_context())


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html", user_email=get_logged_in_email())
@app.route("/responsive")
def responsive_page():
    return render_template("responsive.html", user_email=get_logged_in_email())


# ================= GMAIL ROUTES =================

@app.route("/connect-gmail")
def connect_gmail_route():
    from gmail_reader import connect_gmail
    return connect_gmail()


@app.route("/gmail/callback")
def gmail_callback():
    from gmail_reader import gmail_callback_handler

    gmail_callback_handler()
    return redirect(url_for("gmail_dashboard"))


@app.route("/gmail-dashboard")
def gmail_dashboard():
    from gmail_reader import rank_gmail_emails

    data = rank_gmail_emails()

    print("========== GMAIL DASHBOARD DEBUG ==========")
    print("SUCCESS:", data.get("success"))
    print("MODE:", data.get("mode"))
    print("ERROR:", data.get("error"))
    print("FETCHED:", data.get("fetched_count"))
    print("FILTERED:", data.get("filtered_count"))
    print("SENT TO AI:", data.get("sent_to_ai_count"))
    print("EMAIL ITEMS:", len(data.get("email_items", [])))
    print("===========================================")

    return render_dashboard_with_data(data)


@app.route("/api/gmail/rank")
def api_gmail_rank():
    from gmail_reader import rank_gmail_emails

    data = rank_gmail_emails()

    ranked_output = data.get("ranked_output", {})
    advisor_output = data.get("advisor_output", {})
    display_opportunities = build_display_opportunities(ranked_output, advisor_output)

    return jsonify({
        "success": data.get("success", False),
        "mode": data.get("mode", ""),
        "warning": data.get("warning", ""),
        "note": data.get("note", ""),
        "error": data.get("error", ""),

        "fetched_count": data.get("fetched_count", 0),
        "filtered_count": data.get("filtered_count", 0),
        "sent_to_ai_count": data.get("sent_to_ai_count", 0),

        "email_items": data.get("email_items", []),
        "display_opportunities": display_opportunities,
        "ranked_output": ranked_output,
        "advisor_output": advisor_output
    })


# ================= API ROUTES =================

@app.route("/health", methods=["GET"])
def health():
    user_email = get_logged_in_email()
    profile = safe_profile_by_email(user_email)

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "user_email": user_email,
        "profile_for_logged_in_user": profile
    })


@app.route("/api/profile/me", methods=["GET"])
def api_profile_me():
    try:
        user_email = get_logged_in_email()

        if not user_email:
            return jsonify({
                "success": False,
                "error": "No logged-in email found",
                "profile": None
            }), 401

        profile = get_profile_by_email(user_email)

        return jsonify({
            "success": True,
            "user_email": user_email,
            "profile": profile
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "profile": None
        }), 500


@app.route("/api/profile/save", methods=["POST"])
def api_profile_save():
    try:
        data = request.get_json(force=True)
        user_email = get_logged_in_email()

        account_email = (
            user_email
            or data.get("accountEmail", "")
            or data.get("email", "")
        ).strip().lower()

        if not account_email:
            return jsonify({
                "success": False,
                "error": "Login email is required"
            }), 400

        data["accountEmail"] = account_email

        save_student_profile(data)

        return jsonify({
            "success": True,
            "message": "Profile saved successfully",
            "profile": data
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chatbot_route():
    try:
        data = request.get_json(force=True)
        user_message = (data.get("message") or "").strip()
        user_email = get_logged_in_email()

        if not user_message:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400

        reply = call_chat_safely(user_message, user_email)

        return jsonify({
            "success": True,
            "reply": reply,
            "user_email_used": user_email
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/submit-form", methods=["POST"])
def submit_form():
    try:
        data = request.get_json(force=True)
        user_email = get_logged_in_email()

        account_email = (
            user_email
            or data.get("accountEmail", "")
            or data.get("email", "")
        ).strip().lower()

        if not account_email:
            return jsonify({
                "success": False,
                "error": "Login email is required"
            }), 400

        profile = {
            "accountEmail": account_email,

            "fullName": data.get("fullName", data.get("name", "")),
            "email": data.get("email", account_email),
            "phone": data.get("phone", ""),
            "city": data.get("city", ""),

            "university": data.get("university", ""),
            "degree": data.get("degree", ""),
            "gpa": data.get("gpa", ""),
            "progressType": data.get("progressType", ""),
            "semester": data.get("semester", ""),
            "gradYear": data.get("gradYear", ""),

            "skills": data.get("skills", ""),
            "interests": data.get("interests", ""),

            "certifications": data.get("certifications", ""),
            "experience": data.get("experience", data.get("past_experience", "")),

            "lookingFor": data.get("lookingFor", data.get("preferences", "")),
            "mode": data.get("mode", ""),
            "location": data.get("location", ""),
            "industry": data.get("industry", ""),

            "careerGoals": data.get("careerGoals", ""),

            "linkedin": data.get("linkedin", ""),
            "portfolio": data.get("portfolio", ""),
            "resumeName": data.get("resumeName", ""),

            "financial_need": data.get("financial_need", ""),
            "preferences": data.get("preferences", ""),
            "past_experience": data.get("past_experience", "")
        }

        save_student_profile(profile)

        return jsonify({
            "success": True,
            "message": "Student profile saved successfully",
            "profile": profile
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/process-inputs", methods=["POST"])
def process_inputs_route():
    try:
        user_email = get_logged_in_email()

        text_inputs = []
        image_items = []

        if request.is_json:
            data = request.get_json(force=True)
            text_inputs = data.get("texts", [])[:MAX_INPUTS]
        else:
            text_inputs = request.form.getlist("texts")[:MAX_INPUTS]
            uploaded_files = request.files.getlist("images")[:MAX_INPUTS]
            image_items = process_uploaded_images(uploaded_files)

        text_items = normalize_text_inputs(text_inputs)
        email_items = (text_items + image_items)[:MAX_INPUTS]

        if not email_items:
            return jsonify({
                "success": False,
                "error": "No valid text or image inputs found"
            }), 400

        edu_predictions = []

        for item in email_items:
            pred = predict_email(model, item["sender"])
            edu_predictions.append({
                "sender": item["sender"],
                "subject": item.get("subject", ""),
                "prediction": pred.get("prediction"),
                "confidence": pred.get("confidence"),
                "label": pred.get("label")
            })

        ranked_output = process_emails_safely(email_items, user_email=user_email)
        advisor_output = analyze_opportunities(ranked_output)
        display_opportunities = build_display_opportunities(ranked_output, advisor_output)

        return jsonify({
            "success": True,
            "user_email_used": user_email,

            "input_count": len(email_items),
            "fetched_count": len(email_items),
            "filtered_count": len(email_items),
            "sent_to_ai_count": len(email_items),

            "latest_profile": safe_profile_by_email(user_email),
            "edu_predictions": edu_predictions,

            "email_items": email_items,
            "ranked_output": ranked_output,
            "advisor_output": advisor_output,
            "display_opportunities": display_opportunities
        }), 200

    except Exception as e:
        print("PROCESS INPUTS ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        user_email = get_logged_in_email()
        data = request.get_json(force=True)
        raw_text = (data.get("text") or "").strip()

        if not raw_text:
            return jsonify({
                "success": False,
                "error": "Text is required",
                "opportunities": []
            }), 400

        email_items = normalize_text_inputs([raw_text])

        ranked_output = process_emails_safely(email_items, user_email=user_email)
        advisor_output = analyze_opportunities(ranked_output)
        display_opportunities = build_display_opportunities(ranked_output, advisor_output)

        return jsonify({
            "success": True,
            "user_email_used": user_email,
            "email_items": email_items,
            "opportunities": display_opportunities,
            "ranked_output": ranked_output,
            "advisor_output": advisor_output
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "opportunities": []
        }), 500


@app.route("/detect-edu", methods=["POST"])
def detect_edu_route():
    try:
        data = request.get_json(force=True)
        sender = (data.get("sender") or "").strip()

        if not sender:
            return jsonify({
                "success": False,
                "error": "sender is required"
            }), 400

        result = predict_email(model, sender)

        return jsonify({
            "success": True,
            "result": result
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)