"""Doctor/patient portal and privacy smoke tests."""

from app import app
from database.clinic import DEFAULT_DOCTOR_QUALIFICATION, get_clinic, slot_label
from database.db import get_db


def _login_doctor(client):
    return client.post(
        "/login",
        data={
            "identifier": "doctor@smartskin.demo",
            "password": "Doctor@123",
            "role": "doctor",
        },
        follow_redirects=False,
    )


def test_unauthenticated_dashboards_go_to_login():
    client = app.test_client()
    for path in ("/patient/dashboard", "/doctor/dashboard", "/admin/dashboard"):
        res = client.get(path, follow_redirects=False)
        assert res.status_code == 302
        assert "/login" in (res.headers.get("Location") or "")


def test_patient_login_lands_on_patient_dashboard():
    client = app.test_client()
    res = client.post(
        "/login",
        data={"email": "patient@smartskin.demo", "password": "Patient@123", "role": "patient"},
        follow_redirects=False,
    )
    assert res.status_code == 302
    assert res.headers.get("Location", "").endswith("/patient/dashboard")


def test_doctor_login_uses_email_or_phone_and_password():
    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    assert doctor
    clinic.update_user_profile(int(doctor["id"]), doctor["full_name"], "9876500001")

    client = app.test_client()
    by_email = client.post(
        "/login",
        data={
            "identifier": "doctor@smartskin.demo",
            "password": "Doctor@123",
            "role": "doctor",
        },
        follow_redirects=False,
    )
    assert by_email.status_code == 302
    assert by_email.headers.get("Location", "").endswith("/doctor/dashboard")

    client = app.test_client()
    by_phone = client.post(
        "/login",
        data={
            "identifier": "9876500001",
            "password": "Doctor@123",
            "role": "doctor",
        },
        follow_redirects=False,
    )
    assert by_phone.status_code == 302
    assert by_phone.headers.get("Location", "").endswith("/doctor/dashboard")


def test_hospital_admin_accounts_cannot_sign_in():
    client = app.test_client()
    res = client.post(
        "/login",
        data={
            "email": "admin.kidwai@smartskin.demo",
            "password": "Admin@123",
            "role": "admin",
            "hospital": "Kidwai Memorial Institute of Oncology — Bengaluru",
        },
        follow_redirects=False,
    )
    assert res.status_code == 200
    assert "/admin/dashboard" not in (res.headers.get("Location") or "")


def test_second_hospital_doctor_can_sign_in():
    client = app.test_client()
    res = client.post(
        "/login",
        data={
            "identifier": "doctor.riverside@smartskin.demo",
            "password": "Doctor@123",
            "role": "doctor",
        },
        follow_redirects=False,
    )
    assert res.status_code == 302
    assert "/doctor/dashboard" in (res.headers.get("Location") or "")


def test_doctor_can_self_register_and_admin_sees_roster():
    clinic = get_clinic()
    stamp = str(id(object()))
    email = f"reg.doc.{stamp}@smartskin.demo"
    phone = f"91{stamp[-8:]}"
    client = app.test_client()
    created = client.post(
        "/register",
        data={
            "role": "doctor",
            "full_name": "Dr. Register Test",
            "email": email,
            "phone": phone,
            "password": "Doctor@123",
            "hospital": "Kidwai Memorial Institute of Oncology — Bengaluru",
            "specialization": DEFAULT_DOCTOR_QUALIFICATION,
        },
        follow_redirects=False,
    )
    assert created.status_code == 302
    assert "login" in (created.headers.get("Location") or "")

    signed_in = client.post(
        "/login",
        data={"identifier": phone, "password": "Doctor@123", "role": "doctor"},
        follow_redirects=False,
    )
    assert signed_in.status_code == 302
    assert signed_in.headers.get("Location", "").endswith("/doctor/dashboard")
    client.get("/logout")

    admin = app.test_client()
    admin.post(
        "/login",
        data={"email": "admin@smartskin.demo", "password": "Admin@123", "role": "admin"},
    )
    dash = admin.get("/admin/dashboard")
    body = dash.get_data(as_text=True)
    assert dash.status_code == 200
    assert "Dr. Register Test" in body
    assert "Patients of Dr. Register Test" in body
    assert "All registered patients" in body
    assert "Demo Patient" in body
    user = clinic.get_user_by_email(email)
    assert user and user.get("role") == "doctor"
    clinic.delete_account(int(user["id"]), "doctor")


def test_system_admin_login_does_not_need_hospital():
    client = app.test_client()
    res = client.post(
        "/login",
        data={
            "email": "admin@smartskin.demo",
            "password": "Admin@123",
            "role": "admin",
        },
        follow_redirects=False,
    )
    assert res.status_code == 302
    assert "/admin/dashboard" in (res.headers.get("Location") or "")


def test_home_is_portal_landing():
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Doctor login" in body
    assert "Patient login" in body
    assert "Start New Patient Analysis" not in body
    assert "Pending Cases" not in body
    assert "How it works" in body
    assert "Dermoscopic Analysis" in body
    assert "Meet Our Team" in body
    assert "Prof. Krishna V" in body
    assert "Project Guide" in body
    assert "Khushi M D" in body
    assert "Archana R" in body
    assert "Anusha M" in body
    assert "Vyshnavi R K" in body
    assert "img/team/vyshnavi.jpg" in body
    assert body.index("Anusha M") < body.index("Vyshnavi R K")
    for leak in (
        "56.2",
        "35.8",
        "90.3",
        "PR-AUC",
        "F1-score",
        "confusion matrix",
        "43,000",
        "Recall:",
        "Specificity:",
    ):
        assert leak.lower() not in body.lower()


def test_doctor_dashboard_is_in_consult_not_a_queue():
    client = app.test_client()
    _login_doctor(client)
    res = client.get("/doctor/dashboard")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Start New Patient Analysis" in body
    assert "Today's Analyses" in body
    assert "Pending Cases" not in body
    assert "Work queue" not in body
    assert "New Analysis" in body
    assert "Model Performance" in body


def test_doctor_analytics_shows_combined_cnn_metrics():
    client = app.test_client()
    _login_doctor(client)
    res = client.get("/doctor/analytics")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Combined CNN Models" in body
    assert "EfficientNetB0 + MobileNetV2" in body
    assert "Test Results" in body
    assert "83.42%" in body
    assert "81.46%" in body
    assert "84.90%" in body
    assert "80.08%" in body
    assert "80.76%" in body
    assert "91.3%" in body
    assert "88.8%" in body
    assert "Accuracy" in body
    assert "Sensitivity (Recall)" in body
    assert "Specificity" in body
    assert "Precision" in body
    assert "F1-Score" in body
    assert "ROC-AUC" in body
    assert "PR-AUC" in body
    acc_i = body.find('stat-label">Accuracy')
    rec_i = body.find('stat-label">Sensitivity (Recall)')
    spec_i = body.find('stat-label">Specificity')
    prec_i = body.find('stat-label">Precision')
    f1_i = body.find('stat-label">F1-Score')
    assert 0 <= acc_i < rec_i < spec_i < prec_i < f1_i
    assert "Confusion matrix" in body
    assert "Out of 2,437 images" in body
    assert "1,041 melanoma" in body
    assert "1,396 benign" in body
    assert "of 1,041" in body
    assert "of 1,396" in body
    assert "True Positive (TP)" in body
    assert "True Negative (TN)" in body
    assert "False Positive (FP)" in body
    assert "False Negative (FN)" in body
    assert "848" in body
    assert "1,185" in body
    assert "211" in body
    assert "193" in body
    assert "ROC curve" in body
    assert "Precision–Recall curve" in body
    assert "rocChart" in body
    assert "prChart" in body
    assert "PH2" not in body
    assert "ph2BarChart" not in body
    assert "92.28" not in body
    assert "class-weighted" not in body.lower()
    assert "Frozen Test Split" not in body
    assert "Derm7pt" not in body
    assert "HAM10000" not in body
    assert "accuracyChart" not in body
    assert "86.5%" not in body
    assert "87.3%" not in body
    assert "35.8%" not in body
    assert "56.2%" not in body


def test_patient_dashboard_hides_model_analytics():
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "patient@smartskin.demo", "password": "Patient@123", "role": "patient"},
    )
    res = client.get("/patient/dashboard")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "My Reports" in body
    assert "Book an appointment" in body
    assert "Appointments" in body
    assert "Model Performance" not in body
    assert "Training" not in body
    assert "Melanoma probability" not in body
    assert "Decision threshold" not in body
    assert "AI-assisted result" not in body


def test_patient_report_shows_ajcc_not_model_metrics():
    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    patient = clinic.get_user_by_email("patient@smartskin.demo")
    pred_id = get_db().insert_prediction(
        {
            "patient_id": "P-PAT-AJCC",
            "patient_name": "Demo Patient",
            "prediction": "Malignant",
            "confidence": 0.731,
            "malignant_prob": 0.731,
            "benign_prob": 0.269,
            "model_used": "MobileNet",
            "owner_user_id": patient["id"],
            "assigned_doctor_id": doctor["id"],
            "review_status": "finalized",
            "ajcc_stage": "IIA",
            "t_category": "T2a",
            "n_category": "N0",
            "m_category": "M0",
            "risk_level": "Intermediate",
        }
    )
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "patient@smartskin.demo", "password": "Patient@123", "role": "patient"},
    )
    res = client.get(f"/patient/reports/{pred_id}")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "AJCC stage" in body
    assert "IIA" in body
    assert "T2a" in body
    assert "Clinical stage" in body
    assert "Consulting doctor" in body
    assert "Kidwai Memorial Institute of Oncology" in body
    assert "Dr. Asha Menon" in body
    assert "9876500001" in body
    assert "Doctor phone" in body
    for leak in (
        "Melanoma probability",
        "Decision threshold",
        "Model used",
        "73.1",
        "MobileNet",
        "Grad-CAM",
        "AI-assisted result",
    ):
        assert leak not in body


def test_gradcam_section_omitted_for_non_melanoma():
    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    patient = clinic.get_user_by_email("patient@smartskin.demo")
    pred_id = get_db().insert_prediction(
        {
            "patient_id": "P-UI-BENIGN",
            "patient_name": "Demo Patient",
            "prediction": "Benign",
            "confidence": 0.91,
            "malignant_prob": 0.12,
            "benign_prob": 0.88,
            "model_used": "MobileNet",
            "owner_user_id": patient["id"],
            "assigned_doctor_id": doctor["id"],
            "review_status": "draft",
            "gradcam_path": "static/results/should_not_show.jpg",
        }
    )
    client = app.test_client()
    _login_doctor(client)
    res = client.get(f"/doctor/cases/{pred_id}")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Benign" in body
    assert "AI Explainability" not in body
    assert "Grad-CAM" not in body
    assert "Confidence" in body
    assert "Prediction" in body
    assert "Melanoma probability" not in body
    assert "Decision threshold" not in body


def test_doctor_case_review_reloads_after_saving_prescription():
    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    patient = clinic.get_user_by_email("patient@smartskin.demo")
    pred_id = get_db().insert_prediction(
        {
            "patient_id": "P-RX-RELOAD",
            "patient_name": "Demo Patient",
            "prediction": "Benign",
            "confidence": 0.8,
            "malignant_prob": 0.2,
            "benign_prob": 0.8,
            "model_used": "MobileNet",
            "owner_user_id": patient["id"],
            "assigned_doctor_id": doctor["id"],
            "review_status": "draft",
        }
    )
    client = app.test_client()
    _login_doctor(client)
    saved = client.post(
        f"/doctor/cases/{pred_id}",
        data={
            "action": "save",
            "clinical_impression": "Observe",
            "medicine": "Sunscreen SPF 50",
            "dosage": "Apply",
            "frequency": "Once daily",
            "duration": "4 weeks",
            "instructions": "Morning",
            "prescription_notes": "Use on exposed skin",
        },
        follow_redirects=True,
    )
    assert saved.status_code == 200
    body = saved.get_data(as_text=True)
    assert "TypeError" not in body
    assert "builtin_function_or_method" not in body
    assert "Sunscreen SPF 50" in body
    assert "Prescription and medication" in body


def test_gradcam_section_shown_for_melanoma():
    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    patient = clinic.get_user_by_email("patient@smartskin.demo")
    pred_id = get_db().insert_prediction(
        {
            "patient_id": "P-UI-MEL",
            "patient_name": "Demo Patient",
            "prediction": "Malignant",
            "malignant_prob": 0.68,
            "benign_prob": 0.32,
            "model_used": "MobileNet",
            "owner_user_id": patient["id"],
            "assigned_doctor_id": doctor["id"],
            "review_status": "draft",
            "gradcam_path": "static/uploads/_ui_check.jpg",
        }
    )
    client = app.test_client()
    _login_doctor(client)
    res = client.get(f"/doctor/cases/{pred_id}")
    body = res.get_data(as_text=True)
    assert "Melanoma" in body
    assert "AI Explainability" in body
    assert "Grad-CAM" in body
    assert "Clinical Stage Assessment" in body


def test_admin_dashboard_can_delete_doctor_and_patient():
    clinic = get_clinic()
    stamp = str(id(object()))
    doctor_id = clinic.create_user(
        email=f"delete.doc.{stamp}@smartskin.demo",
        password="Doctor@123",
        role="doctor",
        full_name="Temp Delete Doctor",
        phone=f"81{stamp[-8:]}",
        is_verified=1,
    )
    clinic.upsert_doctor_profile(
        doctor_id,
        hospital="Kidwai Memorial Institute of Oncology — Bengaluru",
        specialization="MD Dermatology",
    )
    patient_id = clinic.create_user(
        email=f"delete.pat.{stamp}@smartskin.demo",
        password="Patient@123",
        role="patient",
        full_name="Temp Delete Patient",
        phone=f"82{stamp[-8:]}",
        is_verified=1,
    )
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "admin@smartskin.demo", "password": "Admin@123", "role": "admin"},
    )
    dash = client.get("/admin/dashboard")
    body = dash.get_data(as_text=True)
    assert "Temp Delete Doctor" in body
    assert "Temp Delete Patient" in body
    assert "Delete" in body

    gone_doc = client.post(
        f"/admin/users/{doctor_id}/delete",
        data={"role": "doctor"},
        follow_redirects=True,
    )
    assert gone_doc.status_code == 200
    after = gone_doc.get_data(as_text=True)
    assert "Temp Delete Doctor was deleted." in after
    assert f'data-bs-target="#doctor-{doctor_id}"' not in after

    gone_pat = client.post(
        f"/admin/users/{patient_id}/delete",
        data={"role": "patient"},
        follow_redirects=True,
    )
    assert gone_pat.status_code == 200
    after = gone_pat.get_data(as_text=True)
    assert "Temp Delete Patient was deleted." in after
    assert f'data-bs-target="#patient-{patient_id}"' not in after

    assert clinic.authenticate("delete.doc." + stamp + "@smartskin.demo", "Doctor@123") is None
    assert clinic.authenticate_patient_phone(f"82{stamp[-8:]}") is None
    remaining = clinic.get_user(doctor_id)
    assert remaining and remaining.get("is_active") == 0


def test_patient_cannot_read_another_patients_report():
    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    other = clinic.get_user_by_email("patient@smartskin.demo")
    pred_id = get_db().insert_prediction(
        {
            "patient_id": "P-OTHER",
            "patient_name": "Other Patient",
            "prediction": "Benign",
            "malignant_prob": 0.1,
            "benign_prob": 0.9,
            "owner_user_id": other["id"],
            "assigned_doctor_id": doctor["id"],
            "review_status": "finalized",
        }
    )
    stranger = clinic.get_user_by_email("stranger.ui@smartskin.demo")
    if stranger:
        stranger_id = int(stranger["id"])
    else:
        stranger_id = clinic.create_user(
            email="stranger.ui@smartskin.demo",
            password="Patient@123",
            role="patient",
            full_name="Stranger Patient",
            phone="9000000001",
            is_verified=1,
        )
        clinic.upsert_patient_profile(stranger_id, assigned_doctor_id=doctor["id"])
    client = app.test_client()
    client.post(
        "/login",
        data={"phone": "9000000001", "role": "patient"},
        follow_redirects=False,
    )
    res = client.get(f"/patient/reports/{pred_id}")
    assert res.status_code in (403, 302)
    body = res.get_data(as_text=True)
    assert "Other Patient" not in body


def test_patient_can_register_search_hospital_and_doctor_can_accept():
    from datetime import date, timedelta

    stamp = str(id(object()))[-8:]
    email = f"book.pat.{stamp}@smartskin.demo"
    phone = f"98{stamp}"
    client = app.test_client()
    created = client.post(
        "/register",
        data={
            "role": "patient",
            "full_name": "Booking Patient",
            "email": email,
            "phone": phone,
            "password": "Patient@123",
            "confirm_password": "Patient@123",
            "sex": "Female",
            "age": "34",
        },
        follow_redirects=False,
    )
    assert created.status_code == 302
    assert "login" in (created.headers.get("Location") or "")

    signed_in = client.post(
        "/login",
        data={"identifier": email, "password": "Patient@123", "role": "patient"},
        follow_redirects=False,
    )
    assert signed_in.status_code == 302
    assert signed_in.headers.get("Location", "").endswith("/patient/dashboard")

    dash = client.get("/patient/dashboard")
    assert dash.status_code == 200
    assert "Book an appointment" in dash.get_data(as_text=True)

    listing = client.get("/patient/appointments")
    assert listing.status_code == 200
    body = listing.get_data(as_text=True)
    assert "Kidwai Memorial Institute of Oncology" in body
    assert "Search hospital" in body

    clinic = get_clinic()
    kidwai = next(
        h
        for h in clinic.list_hospitals()
        if "Kidwai" in (h.get("name") or "")
    )
    doctors_page = client.get(f"/patient/appointments?hospital_id={kidwai['id']}")
    assert doctors_page.status_code == 200
    docs_body = doctors_page.get_data(as_text=True)
    assert "Dr. Asha Menon" in docs_body
    assert "Practises at" in docs_body or "Availability" in docs_body

    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    patient = clinic.get_user_by_email(email)
    day = date.today() + timedelta(days=10)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    taken = set(clinic.taken_slots(int(doctor["id"]), day.isoformat()))
    book_page = client.get(
        f"/patient/appointments?hospital_id={kidwai['id']}&doctor_id={doctor['id']}&date={day.isoformat()}"
    )
    book_body = book_page.get_data(as_text=True)
    assert "9–10" in book_body
    assert "10–11" in book_body
    assert 'type="radio"' in book_body
    assert "select class=\"form-select\"" not in book_body.split("Choose a 1-hour slot", 1)[-1][:1200]
    slot = next(s for s in ("09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00") if s not in taken)
    booked = client.post(
        "/patient/appointments",
        data={
            "hospital_id": kidwai["id"],
            "doctor_id": doctor["id"],
            "appt_date": day.isoformat(),
            "appt_time": slot,
            "reason": "Skin lesion check",
        },
        follow_redirects=False,
    )
    assert booked.status_code == 302
    notice = client.get("/patient/dashboard")
    notice_body = notice.get_data(as_text=True)
    assert "Your appointment request is booked" in notice_body
    assert "Waiting for the doctor to accept" in notice_body

    doctor_client = app.test_client()
    _login_doctor(doctor_client)
    d_dash = doctor_client.get("/doctor/dashboard")
    assert d_dash.status_code == 200
    dash_body = d_dash.get_data(as_text=True)
    assert "Appointment requests" in dash_body
    assert "Booking Patient" in dash_body
    assert "Pending Cases" not in dash_body

    appts = clinic.list_appointments(doctor_id=int(doctor["id"]))
    request_row = next(a for a in appts if int(a["patient_id"]) == int(patient["id"]))
    assert request_row["status"] == "requested"

    accepted = doctor_client.post(
        "/doctor/appointments",
        data={"appt_id": request_row["id"], "status": "accepted"},
        follow_redirects=False,
    )
    assert accepted.status_code == 302
    updated = clinic.get_appointment(int(request_row["id"]))
    assert updated["status"] == "confirmed"
    client.post(
        "/login",
        data={"identifier": email, "password": "Patient@123", "role": "patient"},
        follow_redirects=False,
    )
    booked_home = client.get("/patient/dashboard")
    booked_body = booked_home.get_data(as_text=True)
    assert "Your appointment is booked" in booked_body
    assert "Dr. Asha Menon" in booked_body

    later = day + timedelta(days=1)
    while later.weekday() >= 5:
        later += timedelta(days=1)
    later_taken = set(clinic.taken_slots(int(doctor["id"]), later.isoformat()))
    new_slot = next(
        s for s in ("09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00") if s not in later_taken
    )
    moved = doctor_client.post(
        "/doctor/appointments",
        data={
            "appt_id": request_row["id"],
            "status": "rescheduled",
            "appt_date": later.isoformat(),
            "appt_time": new_slot,
        },
        follow_redirects=False,
    )
    assert moved.status_code == 302
    moved_row = clinic.get_appointment(int(request_row["id"]))
    assert moved_row["status"] == "rescheduled"
    assert moved_row["appt_date"] == later.isoformat()
    assert moved_row["appt_time"] == new_slot
    page = doctor_client.get("/doctor/appointments")
    assert b'name="appt_date"' in page.data
    assert b'name="appt_time"' in page.data


def test_second_patient_cannot_take_same_doctor_slot():
    from datetime import date, timedelta

    clinic = get_clinic()
    doctor = clinic.get_user_by_email("doctor@smartskin.demo")
    stamp = str(id(object()))[-7:]
    day = date.today() + timedelta(days=12)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    taken = set(clinic.taken_slots(int(doctor["id"]), day.isoformat()))
    slot = next(s for s in ("09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00") if s not in taken)

    first = clinic.create_user(
        email=f"slot.a.{stamp}@smartskin.demo",
        password="Patient@123",
        role="patient",
        full_name="First Slot Patient",
        phone=f"91{stamp}1",
        is_verified=1,
    )
    clinic.book_appointment(first, int(doctor["id"]), day.isoformat(), slot, "First booking")

    second_email = f"slot.b.{stamp}@smartskin.demo"
    clinic.create_user(
        email=second_email,
        password="Patient@123",
        role="patient",
        full_name="Second Slot Patient",
        phone=f"91{stamp}2",
        is_verified=1,
    )
    client = app.test_client()
    client.post(
        "/login",
        data={"identifier": second_email, "password": "Patient@123", "role": "patient"},
    )
    kidwai = next(h for h in clinic.list_hospitals() if "Kidwai" in (h.get("name") or ""))
    page = client.get(
        f"/patient/appointments?hospital_id={kidwai['id']}&doctor_id={doctor['id']}&date={day.isoformat()}"
    )
    body = page.get_data(as_text=True)
    assert slot_label(slot) in body
    assert "Booked" in body or "is-taken" in body
    clash = client.post(
        "/patient/appointments",
        data={
            "hospital_id": kidwai["id"],
            "doctor_id": doctor["id"],
            "appt_date": day.isoformat(),
            "appt_time": slot,
            "reason": "Trying the same slot",
        },
        follow_redirects=True,
    )
    assert "not available" in clash.get_data(as_text=True).lower()
    still = clinic.taken_slots(int(doctor["id"]), day.isoformat())
    assert still.count(slot) == 1
