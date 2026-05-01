import os
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

Rules:
1. Give SHORT answers by default.
2. Keep most answers to 3-6 lines unless the user asks for detail, roadmap, steps, full explanation, or examples.
3. If the user asks for more detail, then provide a fuller structured answer.
4. Be practical, clear, and student-friendly.
5. Do not help with illegal, violent, hateful, sexual, or dangerous requests.
6. For mental health, be supportive but do not diagnose.
7. Be forgiving of typos and imperfect English.

If the question is clearly outside scope, reply exactly:
I'm here to help with education, career, wellbeing, and learning. I can't help with that topic.
""".strip()

REFUSAL_MESSAGE = "I'm here to help with education, career, wellbeing, and learning. I can't help with that topic."


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
    }

    for wrong, correct in typo_fixes.items():
        text = text.replace(wrong, correct)

    return text


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

    # Much weaker guardrail:
    # allow most normal study/career/learning/programming/wellbeing questions
    allowed_keywords = [
        "study", "education", "learn", "learning", "student", "school",
        "college", "university", "degree", "semester", "gpa", "exam",
        "subject", "course", "assignment", "quiz", "test", "marks",
        "scholarship", "admission", "deadline", "application", "apply",
        "internship", "fellowship", "research", "bootcamp", "workshop",
        "career", "job", "jobs", "resume", "cv", "skills", "interview",
        "cover letter", "linkedin", "career path", "future", "profession",
        "work", "experience", "mental health", "stress", "burnout",
        "motivation", "anxiety", "focus", "productivity", "confidence",
        "python", "java", "javascript", "c++", "c#", "programming",
        "coding", "developer", "development", "data science", "machine learning",
        "ai", "ml", "software", "web development", "flask", "django",
        "react", "html", "css", "sql", "backend", "frontend", "api",
        "health", "wellbeing", "roadmap", "plan"
    ]

    if any(word in user_input for word in allowed_keywords):
        return True

    # Extra loose fallback so normal questions don't get dismissed
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

    # default: allow unless clearly dangerous
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


def build_user_prompt(user_input: str) -> str:
    profile = get_latest_profile()
    profile_context = build_student_context(profile) if profile else "No student profile available."

    detail_mode = wants_detailed_answer(user_input)

    answer_style = (
        "Give a fuller structured answer with steps, examples, and practical advice."
        if detail_mode
        else "Give a concise answer in 3-6 lines max. Only include the most useful points."
    )

    return f"""
{profile_context}

User question:
{user_input}

Instructions:
- Answer as a helpful student and career mentor.
- {answer_style}
- If the question is about programming or Python, explain it in a learning/career context.
- Be natural, practical, and clear.
- Use the student profile if relevant, but do not force it into every answer.
""".strip()


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
        "max_tokens": 220
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
        "max_tokens": 220
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