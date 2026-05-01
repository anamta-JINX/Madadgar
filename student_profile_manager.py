import json
import os

DATA_FILE = "student_profiles.json"


def load_profiles():
    if not os.path.exists(DATA_FILE):
        return []

    with open(DATA_FILE, "r") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except:
            return []


def save_profiles(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def save_student_profile(profile):
    data = load_profiles()

    email = (
        profile.get("accountEmail")
        or profile.get("email")
        or ""
    ).strip().lower()

    if email:
        data = [
            p for p in data
            if (
                p.get("accountEmail")
                or p.get("email")
                or ""
            ).strip().lower() != email
        ]

    data.append(profile)
    save_profiles(data)

    print("✅ Student profile saved!")


def get_latest_profile():
    data = load_profiles()

    valid_profiles = [
        p for p in data
        if p.get("fullName") or p.get("email") or p.get("degree")
    ]

    if not valid_profiles:
        return None

    return valid_profiles[-1]


def get_profile_by_email(email):
    email = (email or "").strip().lower()

    if not email:
        return get_latest_profile()

    data = load_profiles()

    for profile in reversed(data):
        profile_email = (
            profile.get("accountEmail")
            or profile.get("email")
            or ""
        ).strip().lower()

        if profile_email == email:
            return profile

    return get_latest_profile()


def build_student_context(profile):
    if not profile:
        return "No student profile available."

    return f"""
Student Profile:

Personal Information:
- Full Name: {profile.get('fullName')}
- Email: {profile.get('email')}
- Phone: {profile.get('phone')}
- City / Country: {profile.get('city')}

Academic Details:
- University / College: {profile.get('university')}
- Degree / Field of Study: {profile.get('degree')}
- GPA / CGPA: {profile.get('gpa')}
- Progress Type: {profile.get('progressType')}
- Current Semester / Year: {profile.get('semester')}
- Expected Graduation Year: {profile.get('gradYear')}

Skills & Interests:
- Skills: {profile.get('skills')}
- Interests: {profile.get('interests')}

Certifications & Experience:
- Certifications: {profile.get('certifications')}
- Experience / Projects / Volunteer Work: {profile.get('experience')}

Opportunity Preferences:
- Looking For: {profile.get('lookingFor')}
- Preferred Mode: {profile.get('mode')}
- Preferred Location: {profile.get('location')}
- Preferred Industry: {profile.get('industry')}

Career Goals:
- Career Goals: {profile.get('careerGoals')}

Resume / Portfolio:
- LinkedIn: {profile.get('linkedin')}
- Portfolio / GitHub: {profile.get('portfolio')}
- Resume File: {profile.get('resumeName')}

Use this profile to personalize chatbot advice and rank opportunities.
"""