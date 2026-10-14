"""Doctor clinical dashboard — reuses existing AI pipeline results."""

from __future__ import annotations

import secrets
from datetime import date
from pathlib import Path

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
from database.clinic import (
    CLINIC_HOUR_CHOICES,
    doctor_hour_slots,
    get_clinic,
    normalize_phone_digits,
    parse_availability_hours,
    parse_hour_value,
    slot_label,
)
from reports.pdf_generator import DiagnosticReportGenerator
from reports.prescription_pdf import generate_prescription_pdf
from utils.helpers import generate_unique_id
from utils.metrics_loader import doctor_performance_bundle
from utils.pipeline import pipeline
from models.ensemble import resolve_decision_threshold
from utils.web import (
    clinician_report_fields,
    display_model_name,
    display_prediction,
    is_melanoma,
    to_static_url,
)

bp = Blueprint("doctor", __name__, url_prefix="/doctor")


def _doctor():
    return current_user()


def _require_case(clinic, user, case_id: int):
    record = clinic.db.get_prediction(case_id)
    if not record or not clinic.can_access_case(user, record):
        abort(403)
    return record


def _prescription_items_from_form(form) -> list:
    medicines = form.getlist("medicine")
    dosages = form.getlist("dosage")
    freqs = form.getlist("frequency")
    durs = form.getlist("duration")
    instrs = form.getlist("instructions")
    items = []
    for i, med in enumerate(medicines):
        if not (med or "").strip():
            continue
        items.append(
            {
                "medicine": med.strip(),
                "dosage": dosages[i].strip() if i < len(dosages) else "",
                "frequency": freqs[i].strip() if i < len(freqs) else "",
                "duration": durs[i].strip() if i < len(durs) else "",
                "instructions": instrs[i].strip() if i < len(instrs) else "",
            }
        )
    return items


def _save_case_prescription(clinic, user, record, case_id: int, items, notes: str, diagnosis: str, notify: bool = False):
    patient_id = record.get("owner_user_id")
    if not items or not patient_id:
        return clinic.get_prescription_by_case(case_id)
    code = generate_unique_id("RX")
    rx_id = clinic.upsert_case_prescription(
        case_id=case_id,
        patient_id=int(patient_id),
        doctor_id=user["id"],
        diagnosis=diagnosis,
        notes=notes,
        items=items,
        code=code,
    )
    rx = clinic.get_prescription(rx_id)
    pdf = generate_prescription_pdf(rx, rx["items"])
    clinic.set_prescription_pdf(rx_id, str(pdf), code)
    if notify:
        clinic.notify(
            int(patient_id),
            "A new prescription has been issued by your doctor",
            "Your consultation report includes the medicines you should follow.",
            url_for("patient.report_detail", case_id=case_id),
        )
    clinic.audit(user["id"], "issue_prescription", "prescription", str(rx_id))
    return rx


@bp.route("/dashboard")
@role_required("doctor")
def dashboard():
    clinic = get_clinic()
    user = _doctor()
    stats = clinic.doctor_stats(user["id"])
    recent = clinic.doctor_cases(user["id"])[:8]
    requests = [
        a
        for a in clinic.list_appointments(doctor_id=user["id"])
        if a.get("status") == "requested"
    ]
    return render_template(
        "doctor/dashboard.html",
        stats=stats,
        recent=recent,
        appointment_requests=requests,
    )


@bp.route("/patients/create", methods=["POST"])
@role_required("doctor")
def create_patient():
    clinic = get_clinic()
    user = _doctor()
    full_name = (request.form.get("full_name") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    sex = request.form.get("sex") or "N/A"
    age_raw = (request.form.get("age") or "").strip()
    go_analyze = (request.form.get("next") or "") == "analyze"
    digits = normalize_phone_digits(phone)
    if len(full_name) < 2 or len(digits) < 10:
        flash("Name and a 10-digit phone number are required.", "danger")
        return redirect(url_for("doctor.analyze"))
    age = None
    if age_raw:
        try:
            age = int(age_raw)
        except ValueError:
            flash("Age must be a number.", "danger")
            return redirect(url_for("doctor.analyze"))
    existing = clinic.get_user_by_phone(phone)
    if existing:
        pid = int(existing["id"])
        clinic.upsert_patient_profile(
            pid, assigned_doctor_id=user["id"], age=age, sex=sex
        )
        flash("This phone is already on file. Continue with analysis.", "info")
        return redirect(url_for("doctor.analyze", patient_id=pid))
    email = f"{digits[-10:]}@phone.patient"
    if clinic.get_user_by_email(email):
        email = f"{digits[-10:]}.{secrets.token_hex(3)}@phone.patient"
    pid = clinic.create_user(
        email=email,
        password=secrets.token_urlsafe(16),
        role="patient",
        full_name=full_name,
        phone=phone,
        is_verified=1,
    )
    clinic.upsert_patient_profile(
        pid, age=age, sex=sex, assigned_doctor_id=user["id"],
        chart_id=(request.form.get("chart_id") or "").strip() or None,
    )
    clinic.audit(user["id"], "create_patient", "user", str(pid))
    flash(
        f"Patient saved. They sign in on Patient login with phone {phone}. Next: upload the dermoscopic image.",
        "success",
    )
    if go_analyze:
        return redirect(url_for("doctor.analyze", patient_id=pid))
    return redirect(url_for("doctor.patients"))


@bp.route("/patients")
@role_required("doctor")
def patients():
    clinic = get_clinic()
    user = _doctor()
    q = request.args.get("q") or ""
    rows = clinic.list_doctor_patients(user["id"], search=q)
    return render_template("doctor/patients.html", patients=rows, q=q)


@bp.route("/patients/<int:patient_id>")
@role_required("doctor")
def patient_detail(patient_id: int):
    clinic = get_clinic()
    user = _doctor()
    patient = clinic.get_user(patient_id)
    if not patient or patient.get("role") != "patient":
        abort(404)
    cases = clinic.patient_cases(patient_id)
    if not any(
        c.get("assigned_doctor_id") in (None, user["id"])
        or (patient.get("profile") or {}).get("assigned_doctor_id") == user["id"]
        for c in (cases or [None])
    ) and (patient.get("profile") or {}).get("assigned_doctor_id") not in (user["id"], None):
        # still allow if any case assigned to this doctor
        if not any(c.get("assigned_doctor_id") == user["id"] for c in cases):
            if (patient.get("profile") or {}).get("assigned_doctor_id") != user["id"]:
                abort(403)
    rxs = clinic.list_prescriptions(patient_id=patient_id)
    appts = clinic.list_appointments(patient_id=patient_id)
    return render_template(
        "doctor/patient_detail.html",
        patient=patient,
        cases=cases,
        prescriptions=rxs,
        appointments=appts,
    )


@bp.route("/cases")
@role_required("doctor")
def cases():
    clinic = get_clinic()
    user = _doctor()
    status = request.args.get("status") or ""
    q = request.args.get("q") or ""
    rows = clinic.doctor_cases(user["id"], status=status or None, search=q)
    return render_template("doctor/cases.html", cases=rows, status=status, q=q)


@bp.route("/cases/<int:case_id>", methods=["GET", "POST"])
@role_required("doctor")
def case_review(case_id: int):
    clinic = get_clinic()
    user = _doctor()
    record = _require_case(clinic, user, case_id)
    review = clinic.get_review(case_id) or {}
    owner = clinic.get_user(int(record["owner_user_id"])) if record.get("owner_user_id") else None

    if request.method == "POST":
        action = request.form.get("action") or "save"
        generate = action in {"finalize", "generate"}
        if generate and is_melanoma(record.get("prediction")):
            if not record.get("ajcc_stage"):
                flash(
                    "Enter clinical/pathology values and calculate AJCC stage before generating the report.",
                    "warning",
                )
                return redirect(url_for("doctor.case_review", case_id=case_id) + "#clinical")
        payload = {
            "lesion_location": request.form.get("lesion_location"),
            "lesion_duration": request.form.get("lesion_duration"),
            "symptoms": request.form.get("symptoms"),
            "clinical_observations": request.form.get("clinical_observations")
            or request.form.get("additional_observations"),
            "doctor_notes": request.form.get("doctor_notes")
            or request.form.get("clinical_impression"),
            "ai_agreement": request.form.get("ai_agreement") or "Clinical review recorded",
            "final_diagnosis": request.form.get("final_diagnosis")
            or request.form.get("clinical_impression"),
            "recommendations": request.form.get("recommendations"),
            "follow_up_notes": request.form.get("follow_up_notes"),
            "review_status": "finalized" if generate else "draft",
        }
        clinic.save_review(case_id, user["id"], payload)
        clinic.attach_case_owner(
            case_id,
            record.get("owner_user_id"),
            user["id"],
            review_status=payload["review_status"],
            report_code=record.get("report_code"),
        )
        rx_items = _prescription_items_from_form(request.form)
        diagnosis = (payload.get("final_diagnosis") or display_prediction(record.get("prediction")) or "").strip()
        prescription = _save_case_prescription(
            clinic,
            user,
            record,
            case_id,
            rx_items,
            (request.form.get("prescription_notes") or "").strip(),
            diagnosis,
            notify=generate,
        )
        if generate:
            review_saved = clinic.get_review(case_id)
            notes = payload.get("doctor_notes") or ""
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
                "model_used": display_model_name(record.get("model_used")),
                "decision_threshold": resolve_decision_threshold(),
            }
            patient = {
                "patient_id": record.get("patient_id"),
                "name": record.get("patient_name"),
                "age": record.get("age"),
                "sex": record.get("sex"),
                "phone": (owner.get("phone") if owner else None) or "",
            }
            report_id = record.get("report_code") or generate_unique_id("RPT")
            clinician = clinician_report_fields(user, record.get("doctor_name") or "")
            path = DiagnosticReportGenerator().generate(
                patient=patient,
                prediction=pred,
                staging=staging,
                gradcam_path=record.get("gradcam_path") if is_melanoma(record.get("prediction")) else None,
                lesion_image_path=record.get("image_path"),
                doctor_notes=notes,
                report_id=report_id,
                clinical_review=review_saved,
                verification={
                    "doctor_id": user["id"],
                    "doctor_name": clinician["name"],
                    "hospital": clinician["hospital"],
                    "phone": clinician["phone"],
                    "verified_at": review_saved.get("verified_at") if review_saved else "",
                },
                prescription=prescription,
                clinician=clinician,
            )
            clinic.db.update_prediction(
                case_id,
                {
                    "report_path": str(path),
                    "report_code": report_id,
                    "review_status": "finalized",
                    "assigned_doctor_id": user["id"],
                    "doctor_notes": notes,
                },
            )
            if record.get("owner_user_id"):
                clinic.notify(
                    int(record["owner_user_id"]),
                    "Your report is available",
                    "Your consultation report can be viewed and downloaded.",
                    url_for("patient.report_detail", case_id=case_id),
                )
            clinic.audit(user["id"], "finalize_report", "case", str(case_id))
            flash(
                "Report generated and saved. The patient can view it after signing in with their phone number.",
                "success",
            )
        else:
            flash("Assessment saved.", "success")
        return redirect(url_for("doctor.case_review", case_id=case_id))

    if is_melanoma(record.get("prediction")):
        pipeline.ensure_gradcam(record)
    from gradcam.gradcam import GradCAM

    gradcam_url = (
        to_static_url(record.get("gradcam_path"))
        if is_melanoma(record.get("prediction"))
        else None
    )
    if gradcam_url:
        gradcam_url = f"{gradcam_url}?v={GradCAM.OVERLAY_STYLE_VERSION}"
    images = {
        "original": to_static_url(record.get("image_path")),
        "gradcam": gradcam_url,
    }
    owner = clinic.get_user(int(record["owner_user_id"])) if record.get("owner_user_id") else None
    prescription = clinic.get_prescription_by_case(case_id)
    return render_template(
        "doctor/case_review.html",
        record=record,
        review=review,
        images=images,
        owner=owner,
        prescription=prescription,
        threshold=resolve_decision_threshold(),
        ai_label=display_prediction(record.get("prediction")),
        model_name=display_model_name(record.get("model_used")),
        melanoma=is_melanoma(record.get("prediction")),
    )


@bp.route("/cases/<int:case_id>/staging", methods=["POST"])
@role_required("doctor")
def case_staging(case_id: int):
    clinic = get_clinic()
    user = _doctor()
    record = _require_case(clinic, user, case_id)
    if not is_melanoma(record.get("prediction")):
        flash("AJCC staging applies only when the AI prediction is melanoma.", "warning")
        return redirect(url_for("doctor.case_review", case_id=case_id))
    clinical = {
        "breslow_thickness_mm": request.form.get("breslow_thickness_mm") or 0,
        "ulceration": request.form.get("ulceration", "no"),
        "mitotic_rate": request.form.get("mitotic_rate") or 0,
        "lymph_node_involvement": request.form.get("lymph_node_involvement", "no"),
        "metastasis": request.form.get("metastasis", "no"),
        "num_positive_nodes": request.form.get("num_positive_nodes") or 0,
    }
    try:
        pipeline.apply_ajcc_staging(
            case_id, clinical, doctor_notes=request.form.get("doctor_notes") or ""
        )
        clinic.audit(user["id"], "ajcc_staging", "case", str(case_id))
        flash(
            "Clinical stage calculated from the pathology information you entered. This is not an AI image prediction.",
            "success",
        )
    except Exception as exc:
        flash(str(exc), "danger")
    return redirect(url_for("doctor.case_review", case_id=case_id) + "#clinical")


@bp.route("/performance")
@bp.route("/analytics")
@role_required("doctor")
def performance():
    bundle = doctor_performance_bundle()
    return render_template("doctor/performance.html", bundle=bundle)


@bp.route("/reports")
@role_required("doctor")
def reports():
    clinic = get_clinic()
    user = _doctor()
    rows = [
        c
        for c in clinic.doctor_cases(user["id"])
        if c.get("review_status") == "finalized"
    ]
    return render_template("doctor/reports.html", cases=rows)


@bp.route("/reports/<int:case_id>/download")
@role_required("doctor")
def download_report(case_id: int):
    clinic = get_clinic()
    user = _doctor()
    record = _require_case(clinic, user, case_id)
    path = record.get("report_path")
    if not path or not Path(path).exists():
        flash("PDF unavailable.", "danger")
        return redirect(url_for("doctor.case_review", case_id=case_id))
    p = Path(path)
    return send_file(str(p), as_attachment=True, download_name=p.name)


@bp.route("/prescriptions", methods=["GET", "POST"])
@role_required("doctor")
def prescriptions():
    clinic = get_clinic()
    user = _doctor()
    patients = clinic.list_doctor_patients(user["id"])
    if request.method == "POST":
        try:
            patient_id = int(request.form.get("patient_id") or 0)
            target = clinic.get_user(patient_id)
            if not target or target.get("role") != "patient":
                raise ValueError("Select a valid patient.")
            diagnosis = (request.form.get("diagnosis") or "").strip()
            notes = (request.form.get("notes") or "").strip()
            case_raw = request.form.get("case_id") or ""
            case_id = int(case_raw) if case_raw.isdigit() else None
            medicines = request.form.getlist("medicine")
            dosages = request.form.getlist("dosage")
            freqs = request.form.getlist("frequency")
            durs = request.form.getlist("duration")
            instrs = request.form.getlist("instructions")
            items = []
            for i, med in enumerate(medicines):
                if not med.strip():
                    continue
                items.append(
                    {
                        "medicine": med,
                        "dosage": dosages[i] if i < len(dosages) else "",
                        "frequency": freqs[i] if i < len(freqs) else "",
                        "duration": durs[i] if i < len(durs) else "",
                        "instructions": instrs[i] if i < len(instrs) else "",
                    }
                )
            if not items:
                raise ValueError("Add at least one medicine.")
            code = generate_unique_id("RX")
            rx_id = clinic.create_prescription(
                patient_id=patient_id,
                doctor_id=user["id"],
                diagnosis=diagnosis,
                notes=notes,
                items=items,
                case_id=case_id,
                code=code,
            )
            rx = clinic.get_prescription(rx_id)
            pdf = generate_prescription_pdf(rx, rx["items"])
            clinic.set_prescription_pdf(rx_id, str(pdf), code)
            clinic.notify(
                patient_id,
                "A new prescription has been issued by your doctor",
                "You can view and download your medicines in the patient portal.",
                url_for("patient.prescription_detail", rx_id=rx_id),
            )
            clinic.audit(user["id"], "issue_prescription", "prescription", str(rx_id))
            flash("Prescription issued. It is now visible to the patient.", "success")
            return redirect(url_for("doctor.prescriptions"))
        except ValueError as exc:
            flash(str(exc), "danger")
    rxs = clinic.list_prescriptions(doctor_id=user["id"])
    return render_template(
        "doctor/prescriptions.html", prescriptions=rxs, patients=patients
    )


@bp.route("/appointments", methods=["GET", "POST"])
@role_required("doctor")
def appointments():
    clinic = get_clinic()
    user = _doctor()
    if request.method == "POST":
        appt_id = int(request.form.get("appt_id") or 0)
        status = request.form.get("status") or ""
        appt = clinic.get_appointment(appt_id)
        if not appt or int(appt["doctor_id"]) != int(user["id"]):
            abort(403)
        allowed = {"confirmed", "rescheduled", "completed", "cancelled"}
        if status == "accepted":
            status = "confirmed"
        if status == "declined":
            status = "cancelled"
        if status not in allowed:
            flash("Invalid status.", "danger")
        else:
            try:
                if status == "rescheduled":
                    new_date = (request.form.get("appt_date") or "").strip()
                    new_time = (request.form.get("appt_time") or "").strip()
                    if not new_date or not new_time:
                        raise ValueError("Choose a new date and time to reschedule.")
                    date.fromisoformat(new_date)
                    clinic.reschedule_appointment(appt_id, new_date, new_time)
                    appt = clinic.get_appointment(appt_id) or appt
                else:
                    clinic.set_appointment_status(appt_id, status)
                if status == "confirmed":
                    clinic.upsert_patient_profile(
                        int(appt["patient_id"]), assigned_doctor_id=user["id"]
                    )
                msg = {
                    "confirmed": "Your appointment is booked",
                    "rescheduled": "Your appointment was rescheduled",
                    "completed": "Your appointment was marked completed.",
                    "cancelled": "Your appointment request was declined.",
                }[status]
                clinic.notify(
                    int(appt["patient_id"]),
                    msg,
                    f"{appt.get('doctor_name') or 'Your doctor'}: {appt['appt_date']} at {slot_label(str(appt.get('appt_time') or ''))}.",
                    url_for("patient.appointments"),
                )
                clinic.audit(user["id"], "update_appointment", "appointment", str(appt_id), status)
                flash("Appointment updated.", "success")
            except ValueError as exc:
                flash(str(exc), "danger")
        return redirect(url_for("doctor.appointments"))
    rows = clinic.list_appointments(doctor_id=user["id"])
    pending = [a for a in rows if a.get("status") == "requested"]
    rest = [a for a in rows if a.get("status") != "requested"]
    for row in rest:
        row["taken_times"] = clinic.taken_slots(
            user["id"], str(row.get("appt_date") or ""), exclude_id=int(row["id"])
        )
    return render_template(
        "doctor/appointments.html",
        pending=pending,
        appointments=rest,
        hour_slots=doctor_hour_slots(user),
        today=date.today().isoformat(),
    )


@bp.route("/notifications")
@role_required("doctor")
def notifications():
    clinic = get_clinic()
    user = _doctor()
    items = clinic.list_notifications(user["id"])
    clinic.mark_notifications_read(user["id"])
    return render_template("doctor/notifications.html", items=items)


@bp.route("/profile", methods=["GET", "POST"])
@role_required("doctor")
def profile():
    clinic = get_clinic()
    user = _doctor()
    if request.method == "POST":
        hospital = (request.form.get("hospital") or "").strip()
        if len(hospital) < 2:
            flash("Hospital name is required for doctor sign-in.", "danger")
            user = clinic.get_user(user["id"])
            return render_template(
                "doctor/profile.html",
                user=user,
                hospitals=clinic.list_hospitals(),
                qualifications=clinic.list_qualifications(
                    (user.get("profile") or {}).get("specialization") or ""
                ),
                hour_choices=CLINIC_HOUR_CHOICES,
                hours_from=parse_hour_value(request.form.get("hours_from"), 9),
                hours_until=parse_hour_value(request.form.get("hours_until"), 16),
            )
        specialization = (request.form.get("specialization") or "").strip()
        current_spec = ((user.get("profile") or {}).get("specialization") or "")
        if not clinic.is_allowed_qualification(specialization, current_spec):
            flash("Select a qualification from the list.", "danger")
            user = clinic.get_user(user["id"])
            return render_template(
                "doctor/profile.html",
                user=user,
                hospitals=clinic.list_hospitals(),
                qualifications=clinic.list_qualifications(current_spec),
                hour_choices=CLINIC_HOUR_CHOICES,
                hours_from=parse_hour_value(request.form.get("hours_from"), 9),
                hours_until=parse_hour_value(request.form.get("hours_until"), 16),
            )
        start_hour = parse_hour_value(request.form.get("hours_from"), 9)
        end_hour = parse_hour_value(request.form.get("hours_until"), 16)
        if end_hour <= start_hour:
            flash("Available until must be later than available from.", "danger")
            user = clinic.get_user(user["id"])
            return render_template(
                "doctor/profile.html",
                user=user,
                hospitals=clinic.list_hospitals(),
                qualifications=clinic.list_qualifications(current_spec),
                hour_choices=CLINIC_HOUR_CHOICES,
                hours_from=start_hour,
                hours_until=end_hour,
            )
        clinic.update_user_profile(
            user["id"],
            (request.form.get("full_name") or "").strip(),
            (request.form.get("phone") or "").strip(),
        )
        clinic.upsert_doctor_profile(
            user["id"],
            specialization=specialization,
            hospital=hospital,
            hours_from=f"{start_hour:02d}:00",
            hours_until=f"{end_hour:02d}:00",
            availability=f"{start_hour:02d}:00–{end_hour:02d}:00",
            medical_registration=request.form.get("medical_registration"),
            experience=(request.form.get("experience") or "").strip(),
        )
        flash("Profile saved.", "success")
        return redirect(url_for("doctor.profile"))
    user = clinic.get_user(user["id"])
    profile = user.get("profile") or {}
    hour_start, hour_end = parse_availability_hours(
        str(profile.get("availability") or ""),
        str(profile.get("hours_from") or ""),
        str(profile.get("hours_until") or ""),
    )
    return render_template(
        "doctor/profile.html",
        user=user,
        hospitals=clinic.list_hospitals(),
        qualifications=clinic.list_qualifications(profile.get("specialization") or ""),
        hour_choices=CLINIC_HOUR_CHOICES,
        hours_from=hour_start,
        hours_until=hour_end,
    )


@bp.route("/analyze", methods=["GET", "POST"])
@role_required("doctor")
def analyze():
    """In-consult analysis: patient details, image upload, immediate prediction."""
    from werkzeug.utils import secure_filename
    import config
    from utils.helpers import allowed_file, generate_unique_id, safe_filename

    clinic = get_clinic()
    user = _doctor()
    patients = clinic.list_doctor_patients(user["id"])
    selected_id = request.args.get("patient_id") or request.form.get("patient_id") or ""
    selected = None
    if selected_id and str(selected_id).isdigit():
        selected = clinic.get_user(int(selected_id))
        if selected and selected.get("role") != "patient":
            selected = None
    if request.method == "GET":
        return render_template(
            "doctor/analyze.html",
            patients=patients,
            selected_id=selected_id,
            selected=selected,
        )

    full_name = (request.form.get("full_name") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    chart_id = (request.form.get("chart_id") or "").strip()
    sex = request.form.get("sex") or "N/A"
    age_raw = (request.form.get("age") or "").strip()
    lesion_location = (request.form.get("lesion_location") or "").strip()
    clinical_notes = (request.form.get("clinical_notes") or "").strip()
    digits = normalize_phone_digits(phone)
    if len(full_name) < 2 or len(digits) < 10:
        flash("Patient name and a 10-digit phone number are required.", "danger")
        return redirect(url_for("doctor.analyze", patient_id=selected_id))
    age = None
    if age_raw:
        try:
            age = int(age_raw)
        except ValueError:
            flash("Age must be a number.", "danger")
            return redirect(url_for("doctor.analyze", patient_id=selected_id))

    if "image" not in request.files:
        flash("Upload a dermoscopic image before analyzing.", "danger")
        return redirect(url_for("doctor.analyze", patient_id=selected_id))
    file = request.files["image"]
    if not file or not allowed_file(file.filename or ""):
        flash("Use a JPG, PNG, BMP, TIFF, or WebP image.", "danger")
        return redirect(url_for("doctor.analyze", patient_id=selected_id))

    owner = clinic.get_user_by_phone(phone)
    if owner and owner.get("role") == "patient":
        pid = int(owner["id"])
        clinic.update_user_profile(pid, full_name, phone)
        clinic.upsert_patient_profile(
            pid,
            assigned_doctor_id=user["id"],
            age=age,
            sex=sex,
            chart_id=chart_id or None,
        )
        owner = clinic.get_user(pid)
    else:
        email = f"{digits[-10:]}@phone.patient"
        if clinic.get_user_by_email(email):
            email = f"{digits[-10:]}.{secrets.token_hex(3)}@phone.patient"
        pid = clinic.create_user(
            email=email,
            password=secrets.token_urlsafe(16),
            role="patient",
            full_name=full_name,
            phone=phone,
            is_verified=1,
        )
        clinic.upsert_patient_profile(
            pid,
            age=age,
            sex=sex,
            assigned_doctor_id=user["id"],
            chart_id=chart_id or None,
        )
        clinic.audit(user["id"], "create_patient", "user", str(pid))
        owner = clinic.get_user(pid)

    filename = safe_filename(secure_filename(file.filename))
    save_path = config.UPLOAD_DIR / filename
    file.save(str(save_path))
    profile = owner.get("profile") or {}
    patient = {
        "patient_id": chart_id or profile.get("chart_id") or f"P{owner['id']}",
        "name": owner.get("full_name") or full_name,
        "age": profile.get("age") if profile.get("age") is not None else age,
        "sex": profile.get("sex") or sex,
        "phone": owner.get("phone") or phone,
    }
    try:
        result = pipeline.run(
            image_path=str(save_path),
            patient=patient,
            owner_user_id=int(owner["id"]),
            assigned_doctor_id=user["id"],
        )
    except Exception as exc:
        flash(f"Analysis failed: {exc}", "danger")
        return redirect(url_for("doctor.analyze", patient_id=owner["id"]))

    clinic.save_review(
        result["db_id"],
        user["id"],
        {
            "lesion_location": lesion_location,
            "clinical_observations": clinical_notes,
            "doctor_notes": clinical_notes,
            "review_status": "draft",
        },
    )
    clinic.audit(user["id"], "ai_analyze", "case", str(result["db_id"]))
    pred = str((result.get("prediction") or {}).get("prediction") or "")
    dest = url_for("doctor.case_review", case_id=result["db_id"])
    if is_melanoma(pred):
        flash(
            "AI-assisted prediction: Melanoma. Review Grad-CAM, then enter clinical/pathology information.",
            "info",
        )
        return redirect(dest + "#result")
    flash("AI-assisted prediction complete. Enter your clinical assessment, then generate the report.", "success")
    return redirect(dest)
