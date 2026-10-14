"""Patient portal — view-only medical information plus appointments."""

from __future__ import annotations

from datetime import date, timedelta

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from auth.session import current_user, role_required
from database.clinic import doctor_hour_slots, get_clinic, slot_label
from utils.pipeline import pipeline
from utils.web import clinician_report_fields, is_melanoma, to_static_url

bp = Blueprint("patient", __name__, url_prefix="/patient")


def _patient():
    return current_user()


def _assigned_doctor(clinic, user):
    profile = user.get("profile") or {}
    did = profile.get("assigned_doctor_id")
    if not did:
        docs = clinic.list_verified_doctors()
        return docs[0] if docs else None
    return clinic.get_user(int(did))


@bp.route("/dashboard")
@role_required("patient")
def dashboard():
    clinic = get_clinic()
    user = clinic.get_user(_patient()["id"])
    cases = clinic.patient_cases(user["id"], finalized_only=True)
    latest = cases[0] if cases else None
    appts = clinic.list_appointments(patient_id=user["id"])
    pending = [a for a in appts if a.get("status") == "requested"]
    confirmed = [a for a in appts if a.get("status") in ("confirmed", "rescheduled")]
    notices = clinic.list_notifications(user["id"], unread_only=True)[:8]
    return render_template(
        "patient/dashboard.html",
        user=user,
        cases=cases,
        latest=latest,
        pending_appts=pending,
        confirmed_appts=confirmed,
        notices=notices,
    )


@bp.route("/reports")
@role_required("patient")
def reports():
    clinic = get_clinic()
    user = _patient()
    cases = clinic.patient_cases(user["id"], finalized_only=True)
    return render_template("patient/reports.html", cases=cases)


@bp.route("/reports/<int:case_id>")
@role_required("patient")
def report_detail(case_id: int):
    clinic = get_clinic()
    user = _patient()
    record = clinic.db.get_prediction(case_id)
    if not record or not clinic.can_access_case(user, record):
        abort(403)
    if record.get("review_status") != "finalized":
        abort(404)
    review = clinic.get_review(case_id)
    doctor = None
    if record.get("assigned_doctor_id"):
        doctor = clinic.get_user(int(record["assigned_doctor_id"]))
    if is_melanoma(record.get("prediction")):
        pipeline.ensure_gradcam(record)
    images = {
        "original": to_static_url(record.get("image_path")),
        "gradcam": to_static_url(record.get("gradcam_path"))
        if is_melanoma(record.get("prediction"))
        else None,
    }
    return render_template(
        "patient/report_detail.html",
        record=record,
        review=review,
        doctor=doctor,
        clinician=clinician_report_fields(doctor, record.get("doctor_name") or ""),
        images=images,
        official=True,
        prescription=clinic.get_prescription_by_case(case_id),
    )


@bp.route("/reports/<int:case_id>/download")
@role_required("patient")
def download_report(case_id: int):
    clinic = get_clinic()
    user = _patient()
    record = clinic.db.get_prediction(case_id)
    if not record or not clinic.can_access_case(user, record):
        abort(403)
    if record.get("review_status") != "finalized":
        flash("The official report is available after your doctor finalizes it.", "warning")
        return redirect(url_for("patient.report_detail", case_id=case_id))
    from pathlib import Path

    from models.ensemble import resolve_decision_threshold
    from reports.pdf_generator import DiagnosticReportGenerator

    review = clinic.get_review(case_id)
    doctor = None
    if record.get("assigned_doctor_id"):
        doctor = clinic.get_user(int(record["assigned_doctor_id"]))
    staging = None
    if record.get("ajcc_stage"):
        staging = {
            "stage": record.get("ajcc_stage"),
            "t_category": record.get("t_category"),
            "n_category": record.get("n_category"),
            "m_category": record.get("m_category"),
            "risk_level": record.get("risk_level"),
            "details": record.get("staging_details"),
            "recommendation": record.get("staging_recommendation"),
            "follow_up": record.get("staging_follow_up"),
        }
    pred = {
        "prediction": record.get("prediction"),
        "confidence": record.get("confidence"),
        "malignant_prob": record.get("malignant_prob"),
        "benign_prob": record.get("benign_prob"),
        "model_used": record.get("model_used"),
        "decision_threshold": resolve_decision_threshold(),
    }
    patient = {
        "patient_id": record.get("patient_id"),
        "name": record.get("patient_name"),
        "age": record.get("age"),
        "sex": record.get("sex"),
        "phone": user.get("phone") or "",
    }
    report_id = record.get("report_code") or f"RPT-{record.get('id')}"
    if is_melanoma(record.get("prediction")):
        pipeline.ensure_gradcam(record)
    clinician = clinician_report_fields(doctor, record.get("doctor_name") or "")
    path = DiagnosticReportGenerator().generate(
        patient=patient,
        prediction=pred,
        staging=staging,
        gradcam_path=record.get("gradcam_path") if is_melanoma(record.get("prediction")) else None,
        lesion_image_path=record.get("image_path"),
        doctor_notes=record.get("doctor_notes") or "",
        report_id=report_id,
        clinical_review=review,
        verification={
            "doctor_id": doctor["id"] if doctor else "",
            "doctor_name": clinician["name"],
            "hospital": clinician["hospital"],
            "phone": clinician["phone"],
            "verified_at": review.get("verified_at") if review else "",
        },
        include_model_details=False,
        prescription=clinic.get_prescription_by_case(case_id),
        clinician=clinician,
    )
    p = Path(path)
    if not p.exists():
        flash("PDF file is missing.", "danger")
        return redirect(url_for("patient.report_detail", case_id=case_id))
    return send_file(str(p), as_attachment=True, download_name=p.name)


@bp.route("/appointments", methods=["GET", "POST"])
@role_required("patient")
def appointments():
    clinic = get_clinic()
    user = _patient()
    hospitals = clinic.list_hospitals_for_booking((request.args.get("q") or "").strip())
    hospital_id = 0
    try:
        hospital_id = int(request.args.get("hospital_id") or request.form.get("hospital_id") or 0)
    except ValueError:
        hospital_id = 0
    selected_hospital = clinic.get_hospital(hospital_id) if hospital_id else None
    doctors = clinic.list_doctors_at_hospital(hospital_id) if selected_hospital else []

    if request.method == "POST":
        try:
            doctor_id = int(request.form.get("doctor_id") or 0)
            appt_date = (request.form.get("appt_date") or "").strip()
            appt_time = (request.form.get("appt_time") or "").strip()
            reason = (request.form.get("reason") or "").strip()
            allowed = {int(d["id"]) for d in clinic.list_verified_doctors()}
            if doctor_id not in allowed:
                raise ValueError("Select a listed doctor.")
            if not reason:
                raise ValueError("Describe the reason for the visit.")
            doctor = clinic.get_user(doctor_id)
            allowed_slots = {s["value"] for s in doctor_hour_slots(doctor)}
            if appt_time not in allowed_slots:
                raise ValueError("Select a valid time slot for this doctor.")
            date.fromisoformat(appt_date)
            appt_id = clinic.book_appointment(
                user["id"], doctor_id, appt_date, appt_time, reason
            )
            clinic.notify(
                doctor_id,
                "New appointment request",
                f"{user['full_name']} requested {appt_date} at {slot_label(appt_time)}.",
                url_for("doctor.appointments"),
            )
            clinic.notify(
                user["id"],
                "Your appointment request is booked",
                f"With the doctor on {appt_date} at {slot_label(appt_time)}. Waiting for the doctor to accept.",
                url_for("patient.appointments"),
            )
            clinic.audit(user["id"], "book_appointment", "appointment", str(appt_id))
            flash("Appointment requested. The doctor will accept it when available.", "success")
            return redirect(url_for("patient.appointments"))
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(request.referrer or url_for("patient.appointments"))

    doctor_id = 0
    try:
        doctor_id = int(request.args.get("doctor_id") or 0)
    except ValueError:
        doctor_id = 0
    selected_doctor = next((d for d in doctors if int(d["id"]) == doctor_id), None)
    if selected_doctor is None and doctor_id:
        selected_doctor = next(
            (d for d in clinic.list_verified_doctors() if int(d["id"]) == doctor_id),
            None,
        )
        if selected_doctor and not selected_hospital:
            hid = selected_doctor.get("hospital_id")
            if hid:
                selected_hospital = clinic.get_hospital(int(hid))
                hospital_id = int(hid)
                doctors = clinic.list_doctors_at_hospital(hospital_id)

    appts = clinic.list_appointments(patient_id=user["id"])
    today = date.today().isoformat()
    upcoming = [
        a
        for a in appts
        if str(a.get("appt_date") or "") >= today
        and a.get("status") not in ("cancelled",)
    ]
    past = [
        a
        for a in appts
        if str(a.get("appt_date") or "") < today or a.get("status") in ("completed", "cancelled")
    ]
    dates = []
    d = date.today()
    while len(dates) < 12:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    selected_date = request.args.get("date") or (dates[0] if dates else today)
    taken = (
        clinic.taken_slots(int(selected_doctor["id"]), selected_date)
        if selected_doctor
        else []
    )
    hour_slots = doctor_hour_slots(selected_doctor) if selected_doctor else []
    free = [s["value"] for s in hour_slots if s["value"] not in taken]
    return render_template(
        "patient/appointments.html",
        upcoming=upcoming,
        past=past,
        hospitals=hospitals,
        selected_hospital=selected_hospital,
        doctors=doctors,
        selected_doctor=selected_doctor,
        dates=dates,
        slots=free,
        hour_slots=hour_slots,
        taken=taken,
        selected_date=selected_date,
        query=(request.args.get("q") or "").strip(),
    )


@bp.route("/doctor")
@role_required("patient")
def my_doctor():
    clinic = get_clinic()
    user = _patient()
    doctor = _assigned_doctor(clinic, user)
    return render_template("patient/doctor.html", doctor=doctor, slots=doctor_hour_slots(doctor))


@bp.route("/prescriptions")
@role_required("patient")
def prescriptions():
    clinic = get_clinic()
    user = _patient()
    rxs = clinic.list_prescriptions(patient_id=user["id"])
    return render_template("patient/prescriptions.html", prescriptions=rxs)


@bp.route("/prescriptions/<int:rx_id>")
@role_required("patient")
def prescription_detail(rx_id: int):
    clinic = get_clinic()
    user = _patient()
    rx = clinic.get_prescription(rx_id)
    if not rx or int(rx["patient_id"]) != int(user["id"]):
        abort(403)
    return render_template("patient/prescription_detail.html", rx=rx)


@bp.route("/prescriptions/<int:rx_id>/download")
@role_required("patient")
def download_prescription(rx_id: int):
    clinic = get_clinic()
    user = _patient()
    rx = clinic.get_prescription(rx_id)
    if not rx or int(rx["patient_id"]) != int(user["id"]):
        abort(403)
    from pathlib import Path

    path = rx.get("pdf_path")
    if not path or not Path(path).exists():
        flash("Prescription PDF is not available.", "danger")
        return redirect(url_for("patient.prescription_detail", rx_id=rx_id))
    p = Path(path)
    return send_file(str(p), as_attachment=True, download_name=p.name)


@bp.route("/notifications")
@role_required("patient")
def notifications():
    clinic = get_clinic()
    user = _patient()
    items = clinic.list_notifications(user["id"])
    clinic.mark_notifications_read(user["id"])
    return render_template("patient/notifications.html", items=items)


@bp.route("/profile", methods=["GET", "POST"])
@role_required("patient")
def profile():
    clinic = get_clinic()
    user = _patient()
    if request.method == "POST":
        name = (request.form.get("full_name") or "").strip()
        phone = (request.form.get("phone") or "").strip()
        sex = request.form.get("sex") or "N/A"
        age_raw = request.form.get("age") or ""
        age = int(age_raw) if age_raw.isdigit() else None
        if len(name) < 2:
            flash("Name is required.", "danger")
        else:
            clinic.update_user_profile(user["id"], name, phone)
            clinic.upsert_patient_profile(user["id"], age=age, sex=sex)
            flash("Profile updated.", "success")
            return redirect(url_for("patient.profile"))
    user = clinic.get_user(user["id"])
    return render_template("patient/profile.html", user=user)
