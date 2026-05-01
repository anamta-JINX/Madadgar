import os
import json
import re
from datetime import datetime, date
import requests
from dotenv import load_dotenv

from student_profile_manager import get_latest_profile, build_student_context

# =========================
# LOAD ENV
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL_OPENAI = os.getenv("MODEL_OPENAI", "gpt-4o-mini").strip()
MODEL_GROQ = os.getenv("MODEL_GROQ", "llama-3.3-70b-versatile").strip()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# =========================
# JSON HELPERS
# =========================
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


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, low=0.0, high=10.0):
    return max(low, min(high, value))


def normalize_text(value):
    return str(value or "").strip().lower()


def list_to_text(value):
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value or "")


# =========================
# PROFILE + OPPORTUNITY HELPERS
# =========================
def normalize_opportunity_item(item):
    """
    Keeps compatibility with your existing forward_to_chatbot.py output.
    """
    if not isinstance(item, dict):
        return None

    if "combined_result" in item and isinstance(item["combined_result"], dict):
        cr = item["combined_result"]

        if "combined" in cr and isinstance(cr["combined"], dict):
            combined = cr["combined"].copy()
            combined["_score_data"] = {
                "final_score": cr.get("final_score"),
                "personal_score": cr.get("personal_score"),
                "final_personalized_score": cr.get("final_personalized_score"),
                "is_opportunity": cr.get("is_opportunity")
            }
            return combined

        return cr

    if "combined" in item and isinstance(item["combined"], dict):
        return item["combined"]

    if "original" in item and isinstance(item["original"], dict):
        return item["original"]

    if "title" in item or "summary" in item or "opportunity_type" in item:
        return item

    return None


def get_profile_from_forward_output(forward_output):
    """
    Important:
    This keeps linking the same, but improves logic.
    If forward_to_chatbot.py already used logged-in email,
    it will pass student_profile_used here.
    """
    if isinstance(forward_output, dict):
        profile = (
            forward_output.get("student_profile_used")
            or forward_output.get("student_profile")
            or forward_output.get("profile")
        )

        if isinstance(profile, dict) and profile:
            return profile

    return get_latest_profile()


def build_opportunity_text(opportunity):
    return normalize_text(
        " ".join([
            str(opportunity.get("title", "")),
            str(opportunity.get("opportunity_type", "")),
            str(opportunity.get("summary", "")),
            str(opportunity.get("deadline_found", "")),
            str(opportunity.get("location", "")),
            list_to_text(opportunity.get("benefits", [])),
            list_to_text(opportunity.get("action_items", [])),
            list_to_text(opportunity.get("student_fit_reason", [])),
        ])
    )


# =========================
# DEADLINE / URGENCY LOGIC
# =========================
def extract_deadline_days(deadline_text):
    """
    Best-effort deadline parser.
    It does not replace LLM reasoning, but helps ranking.
    """
    text = str(deadline_text or "").strip()

    if not text or text.lower() in {"not mentioned", "none", "n/a", "unknown"}:
        return None

    today = date.today()

    date_patterns = [
        "%d %B %Y",      # 30 April 2026
        "%d %b %Y",      # 30 Apr 2026
        "%B %d %Y",      # April 30 2026
        "%b %d %Y",      # Apr 30 2026
        "%Y-%m-%d",      # 2026-04-30
        "%d/%m/%Y",      # 30/04/2026
        "%m/%d/%Y",      # 04/30/2026
        "%d-%m-%Y",      # 30-04-2026
        "%m-%d-%Y",      # 04-30-2026
    ]

    cleaned = re.sub(r"[,]", "", text)

    for pattern in date_patterns:
        try:
            parsed = datetime.strptime(cleaned, pattern).date()
            return (parsed - today).days
        except Exception:
            pass

    # Try to capture date-like substring from longer text
    possible_dates = re.findall(
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b|\b[A-Za-z]+\s+\d{1,2}\s+\d{4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        cleaned
    )

    for d in possible_dates:
        for pattern in date_patterns:
            try:
                parsed = datetime.strptime(d.replace(",", ""), pattern).date()
                return (parsed - today).days
            except Exception:
                pass

    return None


def compute_urgency_score(opportunity):
    deadline = opportunity.get("deadline_found", "")
    days_left = extract_deadline_days(deadline)

    opportunity_text = build_opportunity_text(opportunity)

    urgency = safe_float(opportunity.get("urgency_score", 5), 5)

    if days_left is not None:
        if days_left < 0:
            urgency = 1
        elif days_left <= 1:
            urgency = 10
        elif days_left <= 3:
            urgency = 9
        elif days_left <= 7:
            urgency = 8
        elif days_left <= 14:
            urgency = 6.5
        elif days_left <= 30:
            urgency = 5
        else:
            urgency = 3.5

    urgent_words = [
        "deadline",
        "last date",
        "apply now",
        "urgent",
        "closing soon",
        "limited seats",
        "final call",
        "today",
        "tomorrow",
        "this week",
        "shortlisted",
        "interview",
        "confirmation required"
    ]

    for word in urgent_words:
        if word in opportunity_text:
            urgency += 0.5

    return round(clamp(urgency), 2)


# =========================
# ELIGIBILITY LOGIC
# =========================
def extract_gpa_requirement(text):
    text = normalize_text(text)

    patterns = [
        r"gpa\s*(?:of)?\s*(?:at least|minimum|min\.?|>=|above)?\s*([0-5](?:\.\d+)?)",
        r"cgpa\s*(?:of)?\s*(?:at least|minimum|min\.?|>=|above)?\s*([0-5](?:\.\d+)?)",
        r"minimum\s*(?:gpa|cgpa)\s*[:\-]?\s*([0-5](?:\.\d+)?)",
        r"(?:gpa|cgpa)\s*[:\-]?\s*([0-5](?:\.\d+)?)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return safe_float(match.group(1), None)

    return None


def compute_eligibility(profile, opportunity):
    if not profile:
        return {
            "eligible": "Maybe",
            "eligibility_score": 5,
            "matched_criteria": [],
            "missing_criteria": ["No student profile available."],
            "eligibility_notes": "Profile is missing, so eligibility can only be estimated."
        }

    opportunity_text = build_opportunity_text(opportunity)

    matched = []
    missing = []

    student_gpa = safe_float(profile.get("gpa"), None)
    required_gpa = extract_gpa_requirement(opportunity_text)

    if required_gpa is not None:
        if student_gpa is not None and student_gpa >= required_gpa:
            matched.append(f"GPA requirement matched: student GPA {student_gpa} >= required {required_gpa}")
        else:
            missing.append(f"GPA may not meet requirement: required {required_gpa}, student GPA {student_gpa or 'not provided'}")

    degree = normalize_text(profile.get("degree"))
    skills = normalize_text(profile.get("skills"))
    interests = normalize_text(profile.get("interests"))
    looking_for = normalize_text(profile.get("lookingFor"))
    mode = normalize_text(profile.get("mode"))
    location = normalize_text(profile.get("location"))
    industry = normalize_text(profile.get("industry"))
    career_goals = normalize_text(profile.get("careerGoals"))

    profile_words = []

    for field in [degree, skills, interests, looking_for, mode, location, industry, career_goals]:
        profile_words.extend([
            w.strip()
            for w in field.replace(",", " ").replace("/", " ").split()
            if len(w.strip()) > 2
        ])

    keyword_matches = []

    for word in profile_words:
        if word in opportunity_text and word not in keyword_matches:
            keyword_matches.append(word)

    if keyword_matches:
        matched.append("Profile keywords matched: " + ", ".join(keyword_matches[:10]))
    else:
        missing.append("No strong keyword match found between profile and opportunity.")

    if looking_for:
        if looking_for.rstrip("s") in opportunity_text:
            matched.append(f"Opportunity type matches preference: {profile.get('lookingFor')}")
        else:
            missing.append(f"Opportunity may not directly match preference: {profile.get('lookingFor')}")

    if mode:
        if mode in opportunity_text:
            matched.append(f"Preferred mode matched: {profile.get('mode')}")
        elif any(x in opportunity_text for x in ["remote", "hybrid", "on-site", "onsite"]):
            missing.append(f"Mode may not match preferred mode: {profile.get('mode')}")

    if location:
        if location in opportunity_text:
            matched.append(f"Preferred location matched: {profile.get('location')}")
        elif "remote" in opportunity_text:
            matched.append("Remote opportunity can fit location preference.")
        elif any(x in opportunity_text for x in ["location", "city", "country", "on-site", "onsite"]):
            missing.append(f"Location may not match preferred location: {profile.get('location')}")

    eligibility_score = 5

    eligibility_score += min(len(matched) * 1.2, 4)
    eligibility_score -= min(len(missing) * 0.8, 3)

    eligibility_score = round(clamp(eligibility_score), 2)

    if eligibility_score >= 7:
        eligible = "Yes"
    elif eligibility_score >= 4.5:
        eligible = "Maybe"
    else:
        eligible = "No"

    return {
        "eligible": eligible,
        "eligibility_score": eligibility_score,
        "matched_criteria": matched,
        "missing_criteria": missing,
        "eligibility_notes": "Eligibility is estimated using student profile, GPA, keywords, mode, location, and opportunity text."
    }


# =========================
# LOCAL RANKING LOGIC
# =========================
def compute_local_fit_score(profile, opportunity):
    if not profile:
        return 5.0

    opportunity_text = build_opportunity_text(opportunity)

    score = 5.0

    degree = normalize_text(profile.get("degree"))
    skills = normalize_text(profile.get("skills"))
    interests = normalize_text(profile.get("interests"))
    looking_for = normalize_text(profile.get("lookingFor"))
    mode = normalize_text(profile.get("mode"))
    location = normalize_text(profile.get("location"))
    industry = normalize_text(profile.get("industry"))
    career_goals = normalize_text(profile.get("careerGoals"))

    profile_text = " ".join([
        degree,
        skills,
        interests,
        looking_for,
        mode,
        location,
        industry,
        career_goals
    ])

    profile_words = [
        word.strip()
        for word in profile_text.replace(",", " ").replace("/", " ").split()
        if len(word.strip()) > 2
    ]

    for word in profile_words:
        if word in opportunity_text:
            score += 0.35

    opportunity_type = normalize_text(opportunity.get("opportunity_type"))

    if looking_for:
        if looking_for.rstrip("s") in opportunity_text:
            score += 1.5
        if opportunity_type and opportunity_type in looking_for:
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

    if "remote" in opportunity_text and mode == "remote":
        score += 1.2

    if industry and industry in opportunity_text:
        score += 1

    return round(clamp(score), 2)


def compute_final_rank_score(opportunity, advisor_analysis, eligibility_data, profile):
    urgency_score = compute_urgency_score(opportunity)
    fit_score = safe_float(advisor_analysis.get("fit_score"), compute_local_fit_score(profile, opportunity))
    eligibility_score = safe_float(advisor_analysis.get("eligibility_score"), eligibility_data.get("eligibility_score", 5))
    importance_score = safe_float(opportunity.get("importance_score"), 5)
    confidence_score = safe_float(opportunity.get("confidence_score"), 6)

    should_apply = normalize_text(advisor_analysis.get("should_apply"))

    apply_bonus = 0
    if should_apply == "yes":
        apply_bonus = 0.8
    elif should_apply == "maybe":
        apply_bonus = 0.2
    elif should_apply == "no":
        apply_bonus = -1.2

    final_score = (
        urgency_score * 0.30 +
        fit_score * 0.30 +
        eligibility_score * 0.25 +
        importance_score * 0.10 +
        confidence_score * 0.05 +
        apply_bonus
    )

    return {
        "urgency_score": round(clamp(urgency_score), 2),
        "fit_score": round(clamp(fit_score), 2),
        "eligibility_score": round(clamp(eligibility_score), 2),
        "importance_score": round(clamp(importance_score), 2),
        "confidence_score": round(clamp(confidence_score), 2),
        "final_advisor_score": round(clamp(final_score), 2)
    }


# =========================
# PROMPT
# =========================
def build_prompt(opportunity, student_context, eligibility_data, local_fit_score, urgency_score):
    return f"""
You are a highly intelligent student career and opportunity advisor.

You must personalize this opportunity for THIS specific student.

STUDENT CONTEXT:
{student_context}

OPPORTUNITY:
Title: {opportunity.get('title')}
Type: {opportunity.get('opportunity_type')}
Summary: {opportunity.get('summary')}
Deadline: {opportunity.get('deadline_found')}
Location: {opportunity.get('location')}
Benefits: {opportunity.get('benefits')}
Actions: {opportunity.get('action_items')}
Existing Student Fit Reason: {opportunity.get('student_fit_reason')}

LOCAL PRE-CHECK:
Eligibility Estimate:
{json.dumps(eligibility_data, indent=2)}

Local Fit Score: {local_fit_score}/10
Urgency Score: {urgency_score}/10

YOUR JOB:
Analyze:
1. Is the student eligible?
2. How urgent is this opportunity?
3. How well does it match the student's degree, GPA, semester/year, skills, interests, location, mode, and career goals?
4. What exact next steps should the student take?
5. Should the student apply or skip?

IMPORTANT RULES:
- Be strict but fair.
- If deadline has passed, should_apply should usually be "No" unless still useful.
- If eligibility is unclear, use "Maybe" and explain what to verify.
- Do not invent fake requirements.
- If requirements are not visible, say "Not clearly mentioned".
- Return ONLY valid JSON.

Return this exact JSON structure:

{{
  "why_it_matters": "Why this opportunity is valuable or not valuable for this specific student.",
  "eligible": "Yes / Maybe / No",
  "eligibility_score": 0,
  "matched_criteria": [
    "matched criterion"
  ],
  "missing_criteria": [
    "missing or unclear criterion"
  ],
  "requirements": [
    "requirement 1",
    "requirement 2"
  ],
  "urgency_level": "Critical / High / Medium / Low / Expired / Unknown",
  "urgency_reason": "Explain urgency based on deadline and action needed.",
  "fit_score": 0,
  "fit_reason": "Explain the fit score based on the student's profile.",
  "difficulty_level": "Easy / Medium / Hard",
  "estimated_time_to_prepare": "realistic estimate",
  "should_apply": "Yes / Maybe / No",
  "reason_for_decision": "Clear reason why the student should apply, maybe apply, or skip.",
  "next_steps": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "success_tips": [
    "tip 1",
    "tip 2"
  ]
}}
""".strip()


# =========================
# API CALLS
# =========================
def call_openai(prompt):
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_OPENAI,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict student opportunity advisor. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.25
    }

    try:
        res = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)

        if res.status_code != 200:
            print(f"OpenAI API error: {res.status_code} | {res.text}")
            return None

        text = res.json()["choices"][0]["message"]["content"]
        parsed = extract_json_block(text)

        return parsed

    except Exception as e:
        print(f"OpenAI exception: {e}")
        return None


def call_groq(prompt):
    if not GROQ_API_KEY:
        return {"error": "No API keys"}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_GROQ,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict student opportunity advisor. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.25
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)

        if res.status_code != 200:
            return {
                "error": f"API error {res.status_code}",
                "details": res.text
            }

        text = res.json()["choices"][0]["message"]["content"]
        parsed = extract_json_block(text)

        return parsed if parsed else {"error": "Could not parse response"}

    except Exception as e:
        return {"error": str(e)}


# =========================
# FALLBACK ANALYSIS
# =========================
def fallback_analysis(opportunity, eligibility_data, local_fit_score, urgency_score):
    if urgency_score >= 9:
        urgency_level = "Critical"
    elif urgency_score >= 7:
        urgency_level = "High"
    elif urgency_score >= 4:
        urgency_level = "Medium"
    elif urgency_score > 1:
        urgency_level = "Low"
    else:
        urgency_level = "Expired"

    eligible = eligibility_data.get("eligible", "Maybe")

    if eligible == "Yes" and local_fit_score >= 7:
        should_apply = "Yes"
    elif eligible == "No":
        should_apply = "No"
    else:
        should_apply = "Maybe"

    return {
        "why_it_matters": opportunity.get("summary", "This opportunity may be relevant based on the student's profile."),
        "eligible": eligible,
        "eligibility_score": eligibility_data.get("eligibility_score", 5),
        "matched_criteria": eligibility_data.get("matched_criteria", []),
        "missing_criteria": eligibility_data.get("missing_criteria", []),
        "requirements": ["Not clearly mentioned"],
        "urgency_level": urgency_level,
        "urgency_reason": f"Urgency estimated from deadline: {opportunity.get('deadline_found', 'Not mentioned')}",
        "fit_score": local_fit_score,
        "fit_reason": "Fit estimated using profile keywords, preferred opportunity type, mode, location, skills, and goals.",
        "difficulty_level": "Medium",
        "estimated_time_to_prepare": "1-3 days if documents are ready",
        "should_apply": should_apply,
        "reason_for_decision": "Decision estimated using eligibility, fit, and urgency.",
        "next_steps": [
            "Read the official eligibility criteria carefully.",
            "Prepare required documents such as CV, transcript, and statement if needed.",
            "Apply before the deadline or verify if applications are still open."
        ],
        "success_tips": [
            "Customize your CV according to this opportunity.",
            "Highlight matching skills, projects, GPA, and career goals."
        ]
    }


def validate_analysis(analysis, fallback):
    if not isinstance(analysis, dict) or "error" in analysis:
        return fallback

    required_keys = {
        "why_it_matters": fallback["why_it_matters"],
        "eligible": fallback["eligible"],
        "eligibility_score": fallback["eligibility_score"],
        "matched_criteria": fallback["matched_criteria"],
        "missing_criteria": fallback["missing_criteria"],
        "requirements": fallback["requirements"],
        "urgency_level": fallback["urgency_level"],
        "urgency_reason": fallback["urgency_reason"],
        "fit_score": fallback["fit_score"],
        "fit_reason": fallback["fit_reason"],
        "difficulty_level": fallback["difficulty_level"],
        "estimated_time_to_prepare": fallback["estimated_time_to_prepare"],
        "should_apply": fallback["should_apply"],
        "reason_for_decision": fallback["reason_for_decision"],
        "next_steps": fallback["next_steps"],
        "success_tips": fallback["success_tips"]
    }

    for key, value in required_keys.items():
        if key not in analysis or analysis[key] in [None, "", []]:
            analysis[key] = value

    analysis["eligibility_score"] = round(clamp(safe_float(analysis.get("eligibility_score"), fallback["eligibility_score"])), 2)
    analysis["fit_score"] = round(clamp(safe_float(analysis.get("fit_score"), fallback["fit_score"])), 2)

    if analysis.get("eligible") not in ["Yes", "Maybe", "No"]:
        analysis["eligible"] = fallback["eligible"]

    if analysis.get("should_apply") not in ["Yes", "Maybe", "No"]:
        analysis["should_apply"] = fallback["should_apply"]

    if analysis.get("urgency_level") not in ["Critical", "High", "Medium", "Low", "Expired", "Unknown"]:
        analysis["urgency_level"] = fallback["urgency_level"]

    if analysis.get("difficulty_level") not in ["Easy", "Medium", "Hard"]:
        analysis["difficulty_level"] = fallback["difficulty_level"]

    return analysis


# =========================
# MAIN ADVISOR
# =========================
def analyze_opportunities(forward_output):
    """
    Keep this function name the same.
    app.py can still call:
        advisor_output = analyze_opportunities(ranked_output)

    No linking changes needed.
    """
    if not isinstance(forward_output, dict):
        return {
            "ranked_opportunities": [],
            "advisor_results": [],
            "message": "Invalid forward_output format"
        }

    ranked = forward_output.get("ranked_opportunities", [])
    if not isinstance(ranked, list):
        ranked = []

    profile = get_profile_from_forward_output(forward_output)
    student_context = build_student_context(profile) if profile else "No student profile found."

    advisor_results = []

    print("\n=== ANALYZING OPPORTUNITIES FOR SPECIFIC STUDENT ===\n")

    for idx, item in enumerate(ranked):
        combined = normalize_opportunity_item(item)

        if not combined:
            print(f"⚠️ Skipping invalid opportunity at index {idx}")
            continue

        print(f"🔍 Analyzing: {combined.get('title', 'Unknown Opportunity')}")

        eligibility_data = compute_eligibility(profile, combined)
        local_fit_score = compute_local_fit_score(profile, combined)
        urgency_score = compute_urgency_score(combined)

        prompt = build_prompt(
            opportunity=combined,
            student_context=student_context,
            eligibility_data=eligibility_data,
            local_fit_score=local_fit_score,
            urgency_score=urgency_score
        )

        analysis = call_openai(prompt)

        if analysis is None:
            print("⚡ Using Groq fallback...")
            analysis = call_groq(prompt)

        fallback = fallback_analysis(
            opportunity=combined,
            eligibility_data=eligibility_data,
            local_fit_score=local_fit_score,
            urgency_score=urgency_score
        )

        analysis = validate_analysis(analysis, fallback)

        rank_scores = compute_final_rank_score(
            opportunity=combined,
            advisor_analysis=analysis,
            eligibility_data=eligibility_data,
            profile=profile
        )

        analysis.update(rank_scores)

        advisor_results.append({
            "rank": idx + 1,
            "original": combined,
            "student_profile": profile,
            "eligibility_precheck": eligibility_data,
            "local_fit_score": local_fit_score,
            "local_urgency_score": urgency_score,
            "advisor_analysis": analysis
        })

        print("✅ Done\n")

    # Re-rank based on better advisor score
    advisor_results = sorted(
        advisor_results,
        key=lambda x: safe_float(
            x.get("advisor_analysis", {}).get("final_advisor_score"),
            0
        ),
        reverse=True
    )

    for new_rank, item in enumerate(advisor_results, start=1):
        item["rank"] = new_rank

    reranked_opportunities = []

    for item in advisor_results:
        original = item.get("original", {})
        advisor_analysis = item.get("advisor_analysis", {})

        reranked_opportunities.append({
            "combined_result": {
                "final_score": advisor_analysis.get("final_advisor_score", 0),
                "personal_score": advisor_analysis.get("fit_score", 0),
                "final_personalized_score": advisor_analysis.get("final_advisor_score", 0),
                "is_opportunity": True,
                "combined": {
                    **original,
                    "advisor_final_score": advisor_analysis.get("final_advisor_score", 0),
                    "advisor_urgency_score": advisor_analysis.get("urgency_score", 0),
                    "advisor_eligibility_score": advisor_analysis.get("eligibility_score", 0),
                    "advisor_fit_score": advisor_analysis.get("fit_score", 0),
                    "advisor_should_apply": advisor_analysis.get("should_apply", "Maybe"),
                    "advisor_eligible": advisor_analysis.get("eligible", "Maybe"),
                    "advisor_reason": advisor_analysis.get("reason_for_decision", "")
                }
            }
        })

    return {
        "ranked_opportunities": reranked_opportunities,
        "advisor_results": advisor_results,
        "student_profile": profile,
        "total_ranked": len(ranked),
        "total_advised": len(advisor_results),
        "ranking_method": {
            "urgency_weight": "30%",
            "fit_weight": "30%",
            "eligibility_weight": "25%",
            "importance_weight": "10%",
            "confidence_weight": "5%",
            "apply_bonus": "Yes/Maybe/No adjustment"
        }
    }


# =========================
# TEST
# =========================
if __name__ == "__main__":
    sample = {
        "student_profile_used": {
            "fullName": "Anamta",
            "email": "anamta@example.com",
            "degree": "BS Computer Science",
            "gpa": "3.6",
            "semester": "5",
            "skills": "Python, Machine Learning, Web Development",
            "interests": "AI, internships, scholarships",
            "lookingFor": "Internships",
            "mode": "Remote",
            "location": "Pakistan",
            "industry": "Technology",
            "careerGoals": "Become an AI engineer"
        },
        "ranked_opportunities": [
            {
                "combined_result": {
                    "combined": {
                        "title": "AI Internship Program",
                        "opportunity_type": "Internship",
                        "summary": "Remote AI internship for CS students with Python skills.",
                        "deadline_found": "30 April 2026",
                        "location": "Remote",
                        "benefits": ["Mentorship", "Certificate"],
                        "action_items": ["Submit CV", "Apply online"],
                        "urgency_score": 7,
                        "importance_score": 8,
                        "student_fit_score": 8,
                        "confidence_score": 8
                    }
                }
            }
        ]
    }

    result = analyze_opportunities(sample)
    print(json.dumps(result, indent=2))