"""System admin console — one administrator for the whole application."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for

from auth.session import current_user, logout_user, role_required
from database.clinic import DEFAULT_DOCTOR_QUALIFICATION, get_clinic

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _require_system_admin():
    clinic = get_clinic()
    admin = current_user()
    if not clinic.is_system_admin(admin):
        logout_user()
        flash("Only the system administrator can use this console.", "warning")
        return None, None
    return clinic, admin


@bp.route("/dashboard")
@role_required("admin")
def dashboard():
    clinic, admin = _require_system_admin()
    if not admin:
        return redirect(url_for("auth.login", role="admin"))
    doctors = [u for u in clinic.list_users(role="doctor") if u.get("is_active")]
    patients = [u for u in clinic.list_users(role="patient") if u.get("is_active")]
    doctor_by_id = {int(d["id"]): d for d in doctors}
    assigned_ids = set()
    for doctor in doctors:
        roster = clinic.list_doctor_patients(int(doctor["id"]))
        doctor["patients"] = roster
        assigned_ids.update(int(p["id"]) for p in roster)
    for patient in patients:
        assigned_id = (patient.get("profile") or {}).get("assigned_doctor_id")
        try:
            patient["assigned_doctor"] = doctor_by_id.get(int(assigned_id)) if assigned_id else None
        except (TypeError, ValueError):
            patient["assigned_doctor"] = None
    unassigned_patients = [p for p in patients if int(p["id"]) not in assigned_ids]
    stats = clinic.admin_stats(None)
    audit = clinic.list_audit(limit=40, hospital_id=None)
    return render_template(
        "admin/dashboard.html",
        stats=stats,
        doctors=doctors,
        patients=patients,
        unassigned_patients=unassigned_patients,
        audit=audit,
    )


@bp.route("/users/<int:user_id>/verify", methods=["POST"])
@role_required("admin")
def verify_user(user_id: int):
    clinic, admin = _require_system_admin()
    if not admin:
        return redirect(url_for("auth.login", role="admin"))
    target = clinic.get_user(user_id)
    if not target or target.get("role") != "doctor":
        flash("Only doctor accounts can be verified here.", "danger")
        return redirect(url_for("admin.dashboard"))
    flag = 1 if request.form.get("is_verified") == "1" else 0
    clinic.set_user_flags(user_id, is_verified=flag)
    clinic.audit(admin["id"], "verify_user", "user", str(user_id), str(flag))
    flash("User verification updated.", "success")
    return redirect(url_for("admin.dashboard"))


@bp.route("/users/<int:user_id>/delete", methods=["POST"])
@role_required("admin")
def delete_user(user_id: int):
    clinic, admin = _require_system_admin()
    if not admin:
        return redirect(url_for("auth.login", role="admin"))
    role = (request.form.get("role") or "").strip()
    target = clinic.get_user(user_id)
    if not target or target.get("role") not in {"doctor", "patient"} or target.get("role") != role:
        flash("Only a doctor or patient account can be deleted here.", "danger")
        return redirect(url_for("admin.dashboard"))
    name = target.get("full_name") or role
    if clinic.delete_account(user_id, role):
        clinic.audit(admin["id"], "delete_user", role, str(user_id), name)
        flash(f"{name} was deleted.", "success")
    else:
        flash("Could not delete that account.", "danger")
    return redirect(url_for("admin.dashboard"))


@bp.route("/doctors/create", methods=["POST"])
@role_required("admin")
def create_doctor():
    clinic, admin = _require_system_admin()
    if not admin:
        return redirect(url_for("auth.login", role="admin"))
    email = (request.form.get("email") or "").strip().lower()
    name = (request.form.get("full_name") or "").strip()
    password = request.form.get("password") or ""
    specialization = (request.form.get("specialization") or "").strip()
    hospital = (request.form.get("hospital") or "").strip()
    if len(password) < 8 or not email or len(name) < 2:
        flash("Name, email and a password of 8+ characters are required.", "danger")
        return redirect(url_for("admin.dashboard"))
    if len(hospital) < 2:
        flash("Hospital name is required so the doctor can sign in from that hospital.", "danger")
        return redirect(url_for("admin.dashboard"))
    if not clinic.is_allowed_qualification(specialization):
        flash("Select a qualification from the list.", "danger")
        return redirect(url_for("admin.dashboard"))
    if clinic.get_user_by_email(email):
        flash("Email already exists.", "danger")
        return redirect(url_for("admin.dashboard"))
    uid = clinic.create_user(email, password, "doctor", name, is_verified=1, phone=(request.form.get("phone") or "").strip())
    clinic.upsert_doctor_profile(
        uid,
        specialization=specialization or DEFAULT_DOCTOR_QUALIFICATION,
        hospital=hospital,
        availability=request.form.get("availability") or "09:00–16:00",
        hours_from="09:00",
        hours_until="16:00",
        medical_registration=request.form.get("medical_registration") or "",
        experience=(request.form.get("experience") or "").strip(),
    )
    clinic.audit(admin["id"], "create_doctor", "user", str(uid), hospital)
    flash("Doctor account created and verified.", "success")
    return redirect(url_for("admin.dashboard"))


@bp.route("/doctors/<int:user_id>")
@role_required("admin")
def doctor_detail(user_id: int):
    clinic, admin = _require_system_admin()
    if not admin:
        return redirect(url_for("auth.login", role="admin"))
    doctor = clinic.get_user(user_id)
    if not doctor or doctor.get("role") != "doctor":
        flash("Doctor account not found.", "danger")
        return redirect(url_for("admin.dashboard"))
    profile = doctor.get("profile") or {}
    return render_template("admin/doctor_detail.html", doctor=doctor, profile=profile)


@bp.route("/patients/<int:user_id>")
@role_required("admin")
def patient_detail(user_id: int):
    clinic, admin = _require_system_admin()
    if not admin:
        return redirect(url_for("auth.login", role="admin"))
    patient = clinic.get_user(user_id)
    if not patient or patient.get("role") != "patient":
        flash("Patient account not found.", "danger")
        return redirect(url_for("admin.dashboard"))
    profile = patient.get("profile") or {}
    assigned = None
    try:
        if profile.get("assigned_doctor_id"):
            assigned = clinic.get_user(int(profile["assigned_doctor_id"]))
    except (TypeError, ValueError):
        assigned = None
    return render_template(
        "admin/patient_detail.html",
        patient=patient,
        profile=profile,
        assigned=assigned,
    )
