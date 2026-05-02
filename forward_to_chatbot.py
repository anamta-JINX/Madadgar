import os
import json
import re
import requests
from statistics import mean
from dotenv import load_dotenv

from core import load_model, predict_email
from student_profile_manager import (
    get_latest_profile,
    get_profile_by_email,
    build_student_context
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL_OPENAI = os.getenv("MODEL_OPENAI", "gpt-4o-mini").strip()
MODEL_GROQ = os.getenv("MODEL_GROQ", "llama-3.3-70b-versatile").strip()

MODEL_PATH = "edu_detector_model.pkl"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, low=0.0, high=10.0):
    return max(low, min(high, value))


def extract_json_block(text: str):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def extract_sender_from_text(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text or "")
    return match.group(0) if match else ""


def normalize_email_input(item):
    if isinstance(item, str):
        return {
            "sender": extract_sender_from_text(item) or "unknown@edu.com",
            "subject": "",
            "body": item.strip()
        }

    if isinstance(item, dict):
        body = str(item.get("body", "")).strip()
        subject = str(item.get("subject", "")).strip()
        extracted_sender = extract_sender_from_text(body + " " + subject)

        return {
            "sender": str(item.get("sender") or extracted_sender or "unknown@edu.com").strip(),
            "subject": subject,
            "body": body
        }

    return {
        "sender": "unknown@edu.com",
        "subject": "",
        "body": ""
    }


def get_profile_for_user(user_email=""):
    """
    Loads the profile for the logged-in user.
    Falls back to latest profile only if no email/profile is found.
    """
    user_email = (user_email or "").strip().lower()

    if user_email:
        profile = get_profile_by_email(user_email)
        if profile:
            return profile

    return get_latest_profile()


def compute_personal_score(opportunity, profile):
    if not profile:
        return 5.0

    score = 5.0

    opportunity_text = (
        str(opportunity.get("title", "")) + " " +
        str(opportunity.get("summary", "")) + " " +
        str(opportunity.get("opportunity_type", "")) + " " +
        str(opportunity.get("location", "")) + " " +
        " ".join(opportunity.get("benefits", []) if isinstance(opportunity.get("benefits", []), list) else []) + " " +
        " ".join(opportunity.get("action_items", []) if isinstance(opportunity.get("action_items", []), list) else [])
    ).lower()

    degree = str(profile.get("degree", "")).lower()
    skills = str(profile.get("skills", "")).lower()
    interests = str(profile.get("interests", "")).lower()
    looking_for = str(profile.get("lookingFor", "")).lower()
    mode = str(profile.get("mode", "")).lower()
    location = str(profile.get("location", "")).lower()
    industry = str(profile.get("industry", "")).lower()
    career_goals = str(profile.get("careerGoals", "")).lower()

    profile_keywords = " ".join([
        degree,
        skills,
        interests,
        looking_for,
        mode,
        location,
        industry,
        career_goals
    ])

    for word in profile_keywords.replace(",", " ").replace("/", " ").split():
        word = word.strip().lower()

        if len(word) > 2 and word in opportunity_text:
            score += 0.5

    if looking_for and looking_for.rstrip("s") in opportunity_text:
        score += 1.5

    if "scholarship" in opportunity_text and "scholarship" in looking_for:
        score += 2

    if "internship" in opportunity_text and "internship" in looking_for:
        score += 2

    if "job" in opportunity_text and "job" in looking_for:
        score += 2

    if "research" in opportunity_text and "research" in looking_for:
        score += 2

    if mode and mode in opportunity_text:
        score += 1

    if location and location in opportunity_text:
        score += 1

    if industry and industry in opportunity_text:
        score += 1

    return min(score, 10.0)


def build_analysis_prompt(email_record, detector_result, profile):
    student_context = build_student_context(profile) if profile else "No student profile available."

    return f"""
You are analyzing this email for a specific logged-in student.

Use the student's profile to decide:
- whether this email is an opportunity
- how useful it is for this student
- how urgent it is
- whether it matches their degree, skills, interests, goals, location, preferred mode, and opportunity preferences

{student_context}

Email:
Sender: {email_record["sender"]}
Subject: {email_record["subject"]}
Body: {email_record["body"]}

Detector Result:
{json.dumps(detector_result, indent=2)}

Return ONLY valid JSON:

{{
  "is_opportunity": true,
  "opportunity_type": "Scholarship/Internship/Job/Research/Competition/Workshop/Event/Other",
  "title": "...",
  "summary": "...",
  "deadline_found": "...",
  "location": "...",
  "benefits": [],
  "action_items": [],
  "student_fit_reason": [],
  "urgency_score": 0,
  "importance_score": 0,
  "student_fit_score": 0,
  "confidence_score": 0
}}
""".strip()


def call_api(url, key, model, prompt):
    if not key:
        return {"ok": False, "error": "Missing API key"}

    try:
        res = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful student opportunity analyzer. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            },
            timeout=60
        )

        if res.status_code != 200:
            print("API error:", res.status_code, res.text)
            return {"ok": False, "error": res.text}

        data = res.json()
        content = data["choices"][0]["message"]["content"]
        parsed = extract_json_block(content)

        if parsed:
            return {"ok": True, "data": parsed}

        return {"ok": False, "error": "Could not parse JSON"}

    except Exception as e:
        print("API exception:", e)
        return {"ok": False, "error": str(e)}


def combine_model_outputs(openai_result, groq_result, detector_result, profile):
    analyses = []

    if openai_result.get("ok") and isinstance(openai_result.get("data"), dict):
        analyses.append(openai_result["data"])

    if groq_result.get("ok") and isinstance(groq_result.get("data"), dict):
        analyses.append(groq_result["data"])

    if not analyses:
        combined = {
            "is_opportunity": True,
            "title": "Educational Opportunity",
            "opportunity_type": "Other",
            "summary": "Detected opportunity from educational email.",
            "deadline_found": "Not mentioned",
            "location": "Not mentioned",
            "benefits": [],
            "action_items": [],
            "student_fit_reason": [],
            "urgency_score": 5,
            "importance_score": 5,
            "student_fit_score": 5,
            "confidence_score": 5
        }

        personal_score = compute_personal_score(combined, profile)

        return {
            "final_score": 5,
            "personal_score": round(personal_score, 2),
            "final_personalized_score": round((5 * 0.7) + (personal_score * 0.3), 2),
            "consensus": combined["summary"],
            "is_opportunity": True,
            "combined": combined,
            "sources_used": []
        }

    is_opportunity = any(bool(a.get("is_opportunity", False)) for a in analyses)

    if not is_opportunity and detector_result.get("confidence", 0) > 0.4:
        is_opportunity = True

    urgency = clamp(mean([safe_float(a.get("urgency_score", 5)) for a in analyses]))
    importance = clamp(mean([safe_float(a.get("importance_score", 5)) for a in analyses]))
    fit = clamp(mean([safe_float(a.get("student_fit_score", 5)) for a in analyses]))
    confidence = clamp(mean([safe_float(a.get("confidence_score", 5)) for a in analyses]))

    base_score = (urgency + importance + fit + confidence) / 4

    combined = analyses[0]

    combined.setdefault("is_opportunity", is_opportunity)
    combined.setdefault("opportunity_type", "Other")
    combined.setdefault("title", "Untitled Opportunity")
    combined.setdefault("summary", "No summary available.")
    combined.setdefault("deadline_found", "Not mentioned")
    combined.setdefault("location", "Not mentioned")
    combined.setdefault("benefits", [])
    combined.setdefault("action_items", [])
    combined.setdefault("student_fit_reason", [])
    combined.setdefault("urgency_score", urgency)
    combined.setdefault("importance_score", importance)
    combined.setdefault("student_fit_score", fit)
    combined.setdefault("confidence_score", confidence)

    personal_score = compute_personal_score(combined, profile)
    final_score = base_score * 0.7 + personal_score * 0.3

    return {
        "final_score": round(base_score, 2),
        "personal_score": round(personal_score, 2),
        "final_personalized_score": round(final_score, 2),
        "consensus": combined.get("summary", ""),
        "is_opportunity": is_opportunity,
        "combined": combined,
        "sources_used": [
            "openai" if openai_result.get("ok") else None,
            "groq" if groq_result.get("ok") else None
        ]
    }


def process_emails(email_items, user_email=""):
    """
    Main function used by app.py.

    email_items = list of dicts:
    [
        {
            "sender": "...",
            "subject": "...",
            "body": "..."
        }
    ]

    user_email = logged-in email from Flask session.
    This makes ranking personalized to the correct user profile.
    """
    model = load_model(MODEL_PATH)
    profile = get_profile_for_user(user_email)

    results = []
    skipped = []

    for item in email_items:
        email_record = normalize_email_input(item)

        if not email_record["body"] and not email_record["subject"]:
            skipped.append(email_record)
            continue

        detector = predict_email(model, email_record["sender"])

        prompt = build_analysis_prompt(email_record, detector, profile)

        openai_res = call_api(OPENAI_URL, OPENAI_API_KEY, MODEL_OPENAI, prompt)
        groq_res = call_api(GROQ_URL, GROQ_API_KEY, MODEL_GROQ, prompt)

        combined = combine_model_outputs(openai_res, groq_res, detector, profile)

        results.append({
            "sender": email_record["sender"],
            "subject": email_record["subject"],
            "body_preview": email_record["body"][:200],
            "combined_result": combined
        })

    ranked = sorted(
        results,
        key=lambda x: x["combined_result"].get("final_personalized_score", 0),
        reverse=True
    )

    return {
        "ranked_opportunities": ranked,
        "all_processed": results,
        "skipped": skipped,
        "student_profile_used": profile,
        "user_email_used": user_email
    }


if __name__ == "__main__":
    sample_emails = [
        {
            "sender": "scholarships@nust.edu.pk",
            "subject": "Scholarship for CS students",
            "body": "Applications are open for CS students. Deadline is 30 April 2026."
        }
    ]

    output = process_emails(sample_emails, user_email="")
    print(json.dumps(output, indent=2))