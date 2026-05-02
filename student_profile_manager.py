import json
import os
from datetime import datetime

DATA_FILE = "student_profiles.json"


def normalize_email(email):
    return (email or "").strip().lower()


def load_profiles():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception as e:
        print("PROFILE LOAD ERROR:", e)
        return []


def save_profiles(data):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print("PROFILE SAVE ERROR:", e)


def get_profile_email(profile):
    return normalize_email(
        profile.get("accountEmail")
        or profile.get("email")
        or ""
    )


def profile_has_real_data(profile):
    if not isinstance(profile, dict):
        return False

    important_fields = [
        "accountEmail",
        "email",
        "fullName",
        "degree",
        "university",
        "skills",
        "interests",
        "careerGoals"
    ]

    return any(str(profile.get(field, "")).strip() for field in important_fields)


def save_student_profile(profile):
    """
    Saves/updates only the profile for the logged-in email.
    It does NOT overwrite or return another user's profile.
    """

    if not isinstance(profile, dict):
        raise ValueError("Profile must be a dictionary.")

    data = load_profiles()

    email = get_profile_email(profile)

    if not email:
        raise ValueError("Profile must contain accountEmail or email.")

    profile["accountEmail"] = email
    profile["updatedAt"] = datetime.now().isoformat(timespec="seconds")

    if not profile.get("createdAt"):
        profile["createdAt"] = profile["updatedAt"]

    cleaned_profiles = []

    for existing_profile in data:
        existing_email = get_profile_email(existing_profile)

        if not existing_email:
            continue

        if existing_email == email:
            continue

        cleaned_profiles.append(existing_profile)

    cleaned_profiles.append(profile)

    save_profiles(cleaned_profiles)

    print(f"✅ Student profile saved for {email}!")


def get_latest_profile():
    """
    Returns the latest saved profile.

    Use only for debugging/admin purposes.
    Do NOT use this for logged-in user profile loading.
    """

    data = load_profiles()

    valid_profiles = [
        p for p in data
        if profile_has_real_data(p)
    ]

    if not valid_profiles:
        return None

    return valid_profiles[-1]


def get_profile_by_email(email):
    """
    Returns profile ONLY for the requested email.

    Important:
    If a new logged-in email has no profile yet, this returns None.
    It must NOT return get_latest_profile().
    """

    email = normalize_email(email)

    if not email:
        return None

    data = load_profiles()

    for profile in reversed(data):
        profile_email = get_profile_email(profile)

        if profile_email == email:
            return profile

    return None


def delete_profile_by_email(email):
    email = normalize_email(email)

    if not email:
        return False

    data = load_profiles()

    new_data = [
        profile for profile in data
        if get_profile_email(profile) != email
    ]

    if len(new_data) == len(data):
        return False

    save_profiles(new_data)
    return True


def cleanup_profiles():
    """
    Optional helper.
    Removes invalid profiles and duplicate emails.
    Keeps the latest profile for each email.
    """

    data = load_profiles()
    latest_by_email = {}

    for profile in data:
        email = get_profile_email(profile)

        if not email:
            continue

        if not profile_has_real_data(profile):
            continue

        latest_by_email[email] = profile

    cleaned = list(latest_by_email.values())
    save_profiles(cleaned)

    print(f"✅ Cleaned profiles. Total valid profiles: {len(cleaned)}")
    return cleaned


def build_student_context(profile):
    if not profile:
        return "No student profile available."

    return f"""
Student Profile:

Personal Information:
- Full Name: {profile.get('fullName', '')}
- Login Email: {profile.get('accountEmail', '')}
- Email: {profile.get('email', '')}
- Phone: {profile.get('phone', '')}
- City / Country: {profile.get('city', '')}

Academic Details:
- University / College: {profile.get('university', '')}
- Degree / Field of Study: {profile.get('degree', '')}
- GPA / CGPA: {profile.get('gpa', '')}
- Progress Type: {profile.get('progressType', '')}
- Current Semester / Year: {profile.get('semester', '')}
- Expected Graduation Year: {profile.get('gradYear', '')}

Skills & Interests:
- Skills: {profile.get('skills', '')}
- Interests: {profile.get('interests', '')}

Certifications & Experience:
- Certifications: {profile.get('certifications', '')}
- Experience / Projects / Volunteer Work: {profile.get('experience', '')}

Opportunity Preferences:
- Looking For: {profile.get('lookingFor', '')}
- Preferred Mode: {profile.get('mode', '')}
- Preferred Location: {profile.get('location', '')}
- Preferred Industry: {profile.get('industry', '')}

Career Goals:
- Career Goals: {profile.get('careerGoals', '')}

Resume / Portfolio:
- LinkedIn: {profile.get('linkedin', '')}
- Portfolio / GitHub: {profile.get('portfolio', '')}
- Resume File: {profile.get('resumeName', '')}

Use this profile to personalize chatbot advice and rank opportunities.
"""