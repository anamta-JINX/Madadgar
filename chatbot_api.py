import os
import json
import re
import requests
from dotenv import load_dotenv

from student_profile_manager import get_latest_profile, build_student_context

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL_OPENAI = os.getenv("MODEL_OPENAI", "gpt-4o-mini").strip()
MODEL_GROQ = os.getenv("MODEL_GROQ", "llama-3.3-70b-versatile").strip()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """
You are a student-focused AI assistant.

Scope:
- education
- career
- student wellbeing / mental health
- programming, coding, software, and technical skills
- learning roadmaps, study plans, and academic growth
- writing professional student emails (applications, follow-ups, replies)

Rules:
1. Give SHORT answers by default.
2. Keep most answers to 3-6 lines unless the user asks for detail, roadmap, steps, full explanation, or examples.
3. If the user asks for more detail, then provide a fuller structured answer.
4. Be practical, clear, and student-friendly.
5. Do not help with illegal, violent, hateful, sexual, or dangerous requests.
6. For mental health, be supportive but do not diagnose.
7. Be forgiving of typos and imperfect English.
8. When asked to draft an email, produce a ready-to-send email with:
   - Subject line
   - Greeting
   - Short context
   - Clear ask / next steps
   - Polite closing + name

If the question is clearly outside scope, reply exactly:
I'm here to help with education, career, wellbeing, and learning. I can't help with that topic.
""".strip()

REFUSAL_MESSAGE = "I'm here to help with education, career, wellbeing, and learning. I can't help with that topic."


# =========================
# TEXT + JSON HELPERS
# =========================
def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()

    typo_fixes = {
        "carrer": "career",
        "carear": "career",
        "carreer": "career",
        "pyhton": "python",
        "scholorship": "scholarship",
        "intership": "internship",
        "universty": "university",
        "collage": "college",
        "resumee": "resume",
        "email": "email",
        "mail": "email",
        "reply mail": "reply email",
    }

    for wrong, correct in typo_fixes.items():
        text = text.replace(wrong, correct)

    return text


def extract_json_block(text: str):
    """
    Best-effort JSON extractor for when user pastes ranked results.
    """
    if not text:
        return None

    t = text.strip()

    # direct JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # ```json { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # any {...} blob
    match = re.search(r"(\{.*\})", t, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def is_email_request(user_input: str) -> bool:
    t = normalize_text(user_input)
    triggers = [
        "write an email",
        "draft an email",
        "help me write email",
        "help me reply",
        "reply to this email",
        "write a reply",
        "email reply",
        "compose email",
        "send an email",
        "follow up email",
        "follow-up email",
        "write mail",
        "draft mail",
        "professional email",
    ]
    return any(x in t for x in triggers)


def looks_like_ranked_results(obj) -> bool:
    """
    Detects your pipeline output:
    {
      "ranked_opportunities": [...],
      "advisor_results": [...],
      ...
    }
    """
    if not isinstance(obj, dict):
        return False

    if "ranked_opportunities" in obj and isinstance(obj.get("ranked_opportunities"), list):
        return True

    if "advisor_results" in obj and isinstance(obj.get("advisor_results"), list):
        return True

    return False


def summarize_ranked_results_for_email(results_obj, max_items=3) -> str:
    """
    Turn ranked/advisor results JSON into a compact summary that helps the LLM
    draft emails without needing to re-parse a huge blob.
    """
    if not isinstance(results_obj, dict):
        return ""

    ranked = results_obj.get("ranked_opportunities") or []
    advisor_results = results_obj.get("advisor_results") or []

    # Prefer advisor_results because it has next_steps / should_apply etc.
    items = []
    if isinstance(advisor_results, list) and advisor_results:
        for it in advisor_results[:max_items]:
            original = (it or {}).get("original") or {}
            analysis = (it or {}).get("advisor_analysis") or {}
            items.append({
                "title": original.get("title"),
                "type": original.get("opportunity_type"),
                "deadline": original.get("deadline_found"),
                "location": original.get("location"),
                "action_items": original.get("action_items"),
                "benefits": original.get("benefits"),
                "should_apply": analysis.get("should_apply"),
                "reason": analysis.get("reason_for_decision"),
                "next_steps": analysis.get("next_steps"),
            })

    elif isinstance(ranked, list) and ranked:
        # fallback to ranked list
        for it in ranked[:max_items]:
            cr = (it or {}).get("combined_result") or {}
            combined = cr.get("combined") or {}
            items.append({
                "title": combined.get("title"),
                "type": combined.get("opportunity_type"),
                "deadline": combined.get("deadline_found"),
                "location": combined.get("location"),
                "action_items": combined.get("action_items"),
                "benefits": combined.get("benefits"),
                "should_apply": combined.get("advisor_should_apply"),
                "reason": combined.get("advisor_reason"),
                "next_steps": combined.get("action_items"),
            })

    if not items:
        return ""

    lines = ["Ranked opportunities summary (top picks):"]
    for i, x in enumerate(items, start=1):
        lines.append(
            f"{i}) Title: {x.get('title')} | Type: {x.get('type')} | Deadline: {x.get('deadline')} | Location: {x.get('location')}"
        )
        if x.get("should_apply"):
            lines.append(f"   - Suggested: {x.get('should_apply')} | Reason: {x.get('reason')}")
        if x.get("next_steps"):
            lines.append(f"   - Next steps: {x.get('next_steps')}")
        if x.get("action_items"):
            lines.append(f"   - Actions: {x.get('action_items')}")
    return "\n".join(lines).strip()


# =========================
# SAFETY / SCOPE
# =========================
def is_dangerous_question(user_input: str) -> bool:
    user_input = normalize_text(user_input)

    blocked_keywords = [
        "make bomb",
        "build bomb",
        "kill someone",
        "murder",
        "steal password",
        "hack account",
        "carding",
        "fraud",
        "sexual child",
        "rape",
        "buy drugs",
        "make meth",
        "weapon to attack",
    ]

    return any(word in user_input for word in blocked_keywords)


def is_allowed_question(user_input: str) -> bool:
    user_input = normalize_text(user_input)

    if is_dangerous_question(user_input):
        return False

    # allow email writing within student/career scope
    allowed_keywords = [
        "study", "education", "learn", "learning", "student", "school",
        "college", "university", "degree", "semester", "gpa", "exam",
        "subject", "course", "assignment", "quiz", "test", "marks",
        "scholarship", "admission", "deadline", "application", "apply",
        "internship", "fellowship", "research", "bootcamp", "workshop",
        "career", "job", "jobs", "resume", "cv", "skills", "interview",
        "cover letter", "linkedin", "career path", "future", "profession",
        "work", "experience",
        "mental health", "stress", "burnout", "motivation", "anxiety", "focus",
        "productivity", "confidence",
        "python", "java", "javascript", "c++", "c#", "programming", "coding",
        "developer", "development", "data science", "machine learning",
        "ai", "ml", "software", "web development", "flask", "django",
        "react", "html", "css", "sql", "backend", "frontend", "api",
        "health", "wellbeing", "roadmap", "plan",
        "email", "reply", "follow up", "follow-up", "subject line",
    ]

    if any(word in user_input for word in allowed_keywords):
        return True

    helpful_patterns = [
        "tell me about",
        "how do i",
        "how can i",
        "what is",
        "what are",
        "should i",
        "can i learn",
        "guide me",
        "help me with",
        "roadmap",
        "career path",
        "which field",
        "what should i do",
        "how to improve",
    ]

    if any(pattern in user_input for pattern in helpful_patterns):
        return True

    return True


def wants_detailed_answer(user_input: str) -> bool:
    text = normalize_text(user_input)

    detail_triggers = [
        "in detail",
        "detailed",
        "full explanation",
        "step by step",
        "roadmap",
        "complete guide",
        "examples",
        "explain deeply",
        "long answer",
        "full answer",
        "elaborate",
    ]

    return any(trigger in text for trigger in detail_triggers)


# =========================
# PROMPT BUILDING
# =========================
def build_user_prompt(user_input: str) -> str:
    profile = get_latest_profile()
    profile_context = build_student_context(profile) if profile else "No student profile available."

    detail_mode = wants_detailed_answer(user_input)

    # Detect pasted ranked results JSON inside user message
    pasted_json = extract_json_block(user_input)
    ranked_summary = ""
    email_mode = False

    if pasted_json and looks_like_ranked_results(pasted_json):
        ranked_summary = summarize_ranked_results_for_email(pasted_json, max_items=3)

    if is_email_request(user_input) and ranked_summary:
        email_mode = True

    answer_style = (
        "Give a fuller structured answer with steps, examples, and practical advice."
        if detail_mode
        else "Give a concise answer in 3-6 lines max. Only include the most useful points."
    )

    if email_mode:
        answer_style = (
            "Draft a ready-to-send professional email. Keep it concise but complete."
        )

    email_instructions = ""
    if email_mode:
        email_instructions = f"""
EMAIL WRITER MODE:
The user pasted ranked opportunity results. Use them as the factual source.

{ranked_summary}

Email drafting rules:
- Output a Subject line.
- Then the email body.
- Keep it professional and student-friendly.
- Do NOT invent facts (dates, names, acceptance status). If missing, use placeholders like [Recipient Name] and ask 1-3 quick questions at the end.
- If multiple opportunities are present, ask which one they want to email about OR draft separate emails for the top 1-2.
"""

    return f"""
{profile_context}

User message:
{user_input}

Instructions:
- Answer as a helpful student and career mentor.
- {answer_style}
- If the question is about programming or Python, explain it in a learning/career context.
- Be natural, practical, and clear.
- Use the student profile if relevant, but do not force it into every answer.
{email_instructions}
""".strip()


# =========================
# API CALLS
# =========================
def call_openai(user_prompt: str):
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_OPENAI,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 350
    }

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=45)

        if response.status_code != 200:
            print("OpenAI error:", response.status_code, response.text)
            return None

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("OpenAI exception:", e)
        return None


def call_groq(user_prompt: str):
    if not GROQ_API_KEY:
        return "I'm here to help with education, career, wellbeing, and learning. I can't answer right now because no API key is configured."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_GROQ,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 350
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=45)

        if response.status_code != 200:
            print("Groq error:", response.status_code, response.text)
            return "I'm here to help with education, career, wellbeing, and learning. I can't answer right now."

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("Groq exception:", e)
        return "I'm here to help with education, career, wellbeing, and learning. I can't answer right now."


# =========================
# MAIN CHAT
# =========================
def chat(user_input: str) -> str:
    user_input = (user_input or "").strip()

    if not user_input:
        return "Please ask a question."

    if not is_allowed_question(user_input):
        return REFUSAL_MESSAGE

    prompt = build_user_prompt(user_input)

    answer = call_openai(prompt)
    if answer:
        return answer.strip()

    return call_groq(prompt).strip()


if __name__ == "__main__":
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("Bot:", chat(q))