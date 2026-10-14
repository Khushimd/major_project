"""Auth blueprint: login, register, logout, password help."""

from __future__ import annotations

import re
from datetime import date

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from auth.session import current_user, dashboard_endpoint, login_user, logout_user
from database.clinic import DEFAULT_DOCTOR_QUALIFICATION, get_clinic, normalize_phone_digits

bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_LOGIN_ATTEMPTS = 8


def _safe_next(default_endpoint: str):
    nxt = request.args.get("next") or request.form.get("next")
    if nxt and nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return url_for(default_endpoint)


def _expected_role() -> str:
    role = (request.values.get("role") or "patient").strip().lower()
    if role not in {"patient", "doctor", "admin"}:
        return "patient"
    return role


@bp.route("/login", methods=["GET", "POST"])
def login():
    user = current_user()
    if user:
        return redirect(url_for(dashboard_endpoint(user)))

    expected = _expected_role()
    clinic = get_clinic()
    if request.method == "POST":
        attempts = int(session.get("login_attempts") or 0)
        if attempts >= MAX_LOGIN_ATTEMPTS:
            flash("Too many failed sign-in attempts. Close the browser and try later.", "danger")
            return render_template("auth/login.html", expected_role=expected)

        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        phone = (request.form.get("phone") or "").strip()
        identifier = (
            request.form.get("identifier") or email or phone
        ).strip()
        if expected == "patient":
            identifier = (
                request.form.get("identifier") or email or phone
            ).strip()
            if password:
                found = clinic.authenticate_with_password(identifier, password, "patient")
            else:
                found = clinic.authenticate_patient_phone(identifier or phone)
        elif expected == "doctor":
            found = clinic.authenticate_with_password(identifier, password, "doctor")
        else:
            found = clinic.authenticate(email, password)
        if not found:
            session["login_attempts"] = attempts + 1
            clinic.audit(None, "login_failed", "user", identifier or phone or email, "invalid credentials")
            if expected == "doctor":
                flash("Invalid email, phone, or password.", "danger")
            elif expected == "admin":
                flash("Invalid email or password.", "danger")
            else:
                flash("Invalid email, phone, or password.", "danger")
            return render_template("auth/login.html", expected_role=expected)

        if expected == "patient" and found["role"] != "patient":
            flash("This portal is for patients. Please use Doctor login.", "warning")
            return render_template("auth/login.html", expected_role=expected)
        if expected == "doctor" and found["role"] != "doctor":
            flash("This portal is for clinicians. Please use Patient login.", "warning")
            return render_template("auth/login.html", expected_role=expected)
        if expected == "admin" and found["role"] != "admin":
            flash("This portal is for administrators.", "warning")
            return render_template("auth/login.html", expected_role=expected)

        if found["role"] == "doctor" and not found.get("is_verified"):
            flash("This doctor account is awaiting admin verification.", "warning")
            return render_template("auth/login.html", expected_role=expected)

        session.pop("login_attempts", None)
        login_user(found)
        session["last_activity"] = __import__("time").time()
        clinic.audit(found["id"], "login", "user", str(found["id"]))
        dest = dashboard_endpoint(found)
        return redirect(_safe_next(dest))

    return render_template("auth/login.html", expected_role=expected)


@bp.route("/logout")
def logout():
    user = current_user()
    if user:
        get_clinic().audit(user["id"], "logout", "user", str(user["id"]))
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("auth.login"))


def _register_doctor():
    clinic = get_clinic()
    hospitals = clinic.list_hospitals()
    qualifications = clinic.list_qualifications()
    form = request.form if request.method == "POST" else {}
    if request.method != "POST":
        return render_template(
            "auth/register_doctor.html",
            hospitals=hospitals,
            qualifications=qualifications,
        )

    full_name = (form.get("full_name") or "").strip()
    email = (form.get("email") or "").strip().lower()
    phone = (form.get("phone") or "").strip()
    password = form.get("password") or ""
    hospital = (form.get("hospital") or "").strip()
    education = (form.get("specialization") or "").strip()
    digits = normalize_phone_digits(phone)

    errors = []
    if len(full_name) < 2:
        errors.append("Enter your name.")
    if not EMAIL_RE.match(email):
        errors.append("Enter a valid email address.")
    if len(digits) < 10:
        errors.append("Enter a valid 10-digit phone number.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if not clinic.get_or_create_hospital(hospital):
        errors.append("Choose a hospital from the list.")
    if not clinic.is_allowed_qualification(education):
        errors.append("Select your education / qualification from the list.")
    if clinic.get_user_by_email(email):
        errors.append("An account with this email already exists.")
    taken = clinic.get_user_by_phone(phone, role=None)
    if taken and taken.get("is_active"):
        errors.append("An account with this phone number already exists.")

    if errors:
        for err in errors:
            flash(err, "danger")
        return render_template(
            "auth/register_doctor.html",
            hospitals=hospitals,
            qualifications=qualifications,
        )

    uid = clinic.create_user(
        email=email,
        password=password,
        role="doctor",
        full_name=full_name,
        phone=phone,
        is_verified=1,
    )
    clinic.upsert_doctor_profile(
        uid,
        specialization=education or DEFAULT_DOCTOR_QUALIFICATION,
        hospital=hospital,
        availability="09:00–16:00",
        hours_from="09:00",
        hours_until="16:00",
    )
    clinic.audit(uid, "register", "user", str(uid), "doctor self-registration")
    flash("Doctor account created. Sign in with your email or phone and password.", "success")
    return redirect(url_for("auth.login", role="doctor"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for(dashboard_endpoint(current_user())))

    role = (request.values.get("role") or "patient").strip().lower()
    if role == "doctor":
        return _register_doctor()

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        phone = (request.form.get("phone") or "").strip()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        sex = request.form.get("sex") or "N/A"
        dob = (request.form.get("date_of_birth") or "").strip()
        age_raw = (request.form.get("age") or "").strip()

        errors = []
        if len(full_name) < 2:
            errors.append("Enter your full name.")
        if not EMAIL_RE.match(email):
            errors.append("Enter a valid email address.")
        digits = normalize_phone_digits(phone)
        if len(digits) < 10:
            errors.append("Enter a valid 10-digit phone number.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")
        age = None
        if age_raw:
            try:
                age = int(age_raw)
                if age < 0 or age > 120:
                    errors.append("Age is out of range.")
            except ValueError:
                errors.append("Age must be a number.")
        elif dob:
            try:
                born = date.fromisoformat(dob)
                today = date.today()
                age = today.year - born.year - (
                    (today.month, today.day) < (born.month, born.day)
                )
            except ValueError:
                errors.append("Date of birth is invalid.")

        clinic = get_clinic()
        if clinic.get_user_by_email(email):
            errors.append("An account with this email already exists.")
        taken = clinic.get_user_by_phone(phone, role=None)
        if taken and taken.get("is_active"):
            errors.append("An account with this phone number already exists.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return render_template("auth/register.html")

        uid = clinic.create_user(
            email=email,
            password=password,
            role="patient",
            full_name=full_name,
            phone=phone,
            is_verified=1,
        )
        clinic.upsert_patient_profile(
            uid, age=age, sex=sex, date_of_birth=dob or None
        )
        clinic.audit(uid, "register", "user", str(uid), "patient self-registration")
        flash("Profile created. Sign in to search a hospital and book an appointment.", "success")
        return redirect(url_for("auth.login", role="patient"))

    return render_template("auth/register.html")


@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        flash(
            "Password resets are handled by the clinic administrator. "
            "Contact your care team with the email on this account.",
            "info",
        )
        return redirect(url_for("auth.login", role="patient"))
    return render_template("auth/forgot.html")
