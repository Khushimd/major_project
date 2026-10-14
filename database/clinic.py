"""Clinic layer: users, appointments, prescriptions, notifications, audit."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash

from database.db import Database, get_db
from utils.logger import setup_logger

logger = setup_logger("clinic")

CLINIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    full_name TEXT NOT NULL,
    phone TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    is_verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    user_id INTEGER PRIMARY KEY,
    age INTEGER,
    sex TEXT,
    address TEXT,
    date_of_birth TEXT,
    assigned_doctor_id INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(assigned_doctor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS hospitals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,
    address TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doctor_profiles (
    user_id INTEGER PRIMARY KEY,
    specialization TEXT,
    hospital TEXT,
    hospital_id INTEGER,
    availability TEXT,
    medical_registration TEXT,
    experience TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(hospital_id) REFERENCES hospitals(id)
);

CREATE TABLE IF NOT EXISTS admin_profiles (
    user_id INTEGER PRIMARY KEY,
    hospital TEXT,
    hospital_id INTEGER,
    is_system INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(hospital_id) REFERENCES hospitals(id)
);

CREATE TABLE IF NOT EXISTS clinical_reviews (
    case_id INTEGER PRIMARY KEY,
    lesion_location TEXT,
    lesion_duration TEXT,
    symptoms TEXT,
    clinical_observations TEXT,
    doctor_notes TEXT,
    ai_agreement TEXT,
    final_diagnosis TEXT,
    recommendations TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending_review',
    doctor_id INTEGER,
    verified_at TEXT,
    FOREIGN KEY(case_id) REFERENCES predictions(id),
    FOREIGN KEY(doctor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    appt_date TEXT NOT NULL,
    appt_time TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'requested',
    created_at TEXT NOT NULL,
    FOREIGN KEY(patient_id) REFERENCES users(id),
    FOREIGN KEY(doctor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS prescriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    case_id INTEGER,
    diagnosis TEXT,
    notes TEXT,
    pdf_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(patient_id) REFERENCES users(id),
    FOREIGN KEY(doctor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS prescription_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prescription_id INTEGER NOT NULL,
    medicine TEXT NOT NULL,
    dosage TEXT,
    frequency TEXT,
    duration TEXT,
    instructions TEXT,
    FOREIGN KEY(prescription_id) REFERENCES prescriptions(id)
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    link TEXT,
    is_read INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    entity TEXT,
    entity_id TEXT,
    details TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_hospitals_key ON hospitals(name_key);
CREATE INDEX IF NOT EXISTS idx_appt_patient ON appointments(patient_id);
CREATE INDEX IF NOT EXISTS idx_appt_doctor ON appointments(doctor_id);
CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id);
"""

PRED_EXTRA_COLS = {
    "owner_user_id": "ALTER TABLE predictions ADD COLUMN owner_user_id INTEGER",
    "assigned_doctor_id": "ALTER TABLE predictions ADD COLUMN assigned_doctor_id INTEGER",
    "review_status": "ALTER TABLE predictions ADD COLUMN review_status TEXT",
    "report_code": "ALTER TABLE predictions ADD COLUMN report_code TEXT",
}

APPOINTMENT_SLOTS = ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00"]
CLINIC_HOUR_CHOICES = list(range(8, 21))


def _clock_hour(hour: int) -> str:
    hour = int(hour) % 24
    if hour == 0:
        return "12"
    if hour <= 12:
        return str(hour)
    return str(hour - 12)


def slot_label(value: str) -> str:
    hour = parse_hour_value(value, 9)
    return f"{_clock_hour(hour)}–{_clock_hour(hour + 1)}"


def parse_hour_value(value: str, default: int = 9) -> int:
    match = re.search(r"(\d{1,2})", str(value or ""))
    if not match:
        return default
    hour = int(match.group(1))
    if hour == 24:
        hour = 0
    return hour if 0 <= hour <= 23 else default


def parse_availability_hours(
    availability: str = "", hours_from: str = "", hours_until: str = ""
) -> tuple[int, int]:
    if (hours_from or "").strip() and (hours_until or "").strip():
        start = parse_hour_value(hours_from, 9)
        end = parse_hour_value(hours_until, 16)
    else:
        found = [int(h) for h, _ in re.findall(r"(\d{1,2})(?::(\d{2}))?", availability or "")]
        found = [h for h in found if 0 <= h <= 23]
        if len(found) >= 2:
            start, end = found[0], found[1]
        elif found:
            start, end = found[0], 16
        else:
            start, end = 9, 16
    if end <= start:
        end = min(23, start + 1)
    return start, end


def hour_slots_for_range(start: int, end: int) -> List[Dict[str, str]]:
    return [
        {"value": f"{hour:02d}:00", "label": f"{_clock_hour(hour)}–{_clock_hour(hour + 1)}"}
        for hour in range(int(start), int(end))
    ]


def doctor_hour_slots(doctor: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    doctor = doctor or {}
    profile = doctor.get("profile") or {}
    start, end = parse_availability_hours(
        str(doctor.get("availability") or profile.get("availability") or ""),
        str(doctor.get("hours_from") or profile.get("hours_from") or ""),
        str(doctor.get("hours_until") or profile.get("hours_until") or ""),
    )
    return hour_slots_for_range(start, end)

# Karnataka centres that diagnose cancer (biopsy, pathology, imaging) and treat it
# (surgical / medical / radiation oncology). Not a general hospital directory.
KNOWN_CANCER_HOSPITALS = (
    ("Kidwai Memorial Institute of Oncology — Bengaluru", "Dr. M.H. Marigowda Road, Bengaluru, Karnataka 560029"),
    ("HCG Cancer Centre — Kalinga Rao Road, Bengaluru", "No. 8, HCG Towers, P. Kalinga Rao Road, Bengaluru, Karnataka 560027"),
    ("HCG Cancer Centre — Double Road, Bengaluru", "No. 44-45/2, 2nd Cross, Raja Ram Mohan Roy Extension, Bengaluru 560027"),
    ("HCG MSR Gate — Bengaluru", "MSR Nagar, New BEL Road, Bengaluru, Karnataka 560054"),
    ("Cytecare Cancer Hospitals — Yelahanka, Bengaluru", "Near Venkatala, Bagalur Main Road, Yelahanka, Bengaluru, Karnataka 560064"),
    ("Sri Shankara Cancer Hospital — Basavanagudi, Bengaluru", "1st Cross, Shankarapuram, Basavanagudi, Bengaluru, Karnataka 560004"),
    ("Mazumdar Shaw Cancer Center — Narayana Health City, Bengaluru", "258/A, Bommasandra Industrial Area, Hosur Road, Bengaluru 560099"),
    ("Vydehi Institute of Oncology — Whitefield, Bengaluru", "82, EPIP Area, Nallurhalli, Whitefield, Bengaluru, Karnataka 560066"),
    ("Manipal Comprehensive Cancer Centre — Old Airport Road, Bengaluru", "98, HAL Old Airport Road, Bengaluru, Karnataka 560017"),
    ("Manipal Hospital — Whitefield, Bengaluru", "143, 212-215, EPIP Industrial Area, Whitefield, Bengaluru 560066"),
    ("Apollo Cancer Centre — Bannerghatta Road, Bengaluru", "154/11, Opposite IIM, Bannerghatta Road, Bengaluru, Karnataka 560076"),
    ("Fortis Cancer Institute — Bannerghatta Road, Bengaluru", "154/9, Bannerghatta Road, Opposite IIM-B, Bengaluru, Karnataka 560076"),
    ("St. John's Medical College Hospital — Oncology, Bengaluru", "Sarjapur Road, Koramangala, Bengaluru, Karnataka 560034"),
    ("Sakra World Hospital — Oncology, Bengaluru", "52/2 & 52/3, Devarabeesanahalli, Outer Ring Road, Bengaluru 560103"),
    ("Aster International Institute of Oncology — CMI Hebbal, Bengaluru", "No. 43/2, New Airport Road, NH 7, Hebbal, Bengaluru 560092"),
    ("Aster Whitefield Hospital — Oncology, Bengaluru", "Plot No. 2, 3 & 4, Sadaramangala, Whitefield, Bengaluru 560066"),
    ("Ramaiah Memorial Hospital — Cancer Centre, Bengaluru", "New BEL Road, MSR Nagar, Bengaluru, Karnataka 560054"),
    ("Bangalore Baptist Hospital — Oncology, Bengaluru", "Bellary Road, Hebbal, Bengaluru, Karnataka 560024"),
    ("BGS Gleneagles Global Hospital — Oncology, Kengeri, Bengaluru", "67, Uttarahalli Road, Kengeri, Bengaluru, Karnataka 560060"),
    ("Bharath Hospital & Institute of Oncology — Mysuru", "Hebbal Industrial Area, Mysuru, Karnataka 570016"),
    ("JSS Hospital — Oncology, Mysuru", "Mahatma Gandhi Road, Mysuru, Karnataka 570004"),
    ("Apollo BGS Hospitals — Oncology, Mysuru", "Adichunchanagiri Road, Kuvempunagar, Mysuru 570023"),
    ("Kasturba Hospital — Oncology, Manipal", "Madhav Nagar, Manipal, Udupi District, Karnataka 576104"),
    ("KMC Hospital — Oncology, Attavar, Mangaluru", "Attavar, Mangaluru, Karnataka 575001"),
    ("Father Muller Medical College Hospital — Oncology, Mangaluru", "Father Muller Road, Kankanady, Mangaluru, Karnataka 575002"),
    ("A.J. Hospital & Research Centre — Oncology, Mangaluru", "Kuntikana, Mangaluru, Karnataka 575004"),
    ("Yenepoya Medical College Hospital — Oncology, Mangaluru", "University Road, Deralakatte, Mangaluru, Karnataka 575018"),
    ("Justice K.S. Hegde Charitable Hospital — Oncology, Mangaluru", "Nitte University, Deralakatte, Mangaluru, Karnataka 575018"),
    ("KLE Dr. Prabhakar Kore Hospital — Oncology, Belagavi", "Nehru Nagar, Belagavi, Karnataka 590010"),
    ("HCG NMR Cancer Centre — Belagavi", "Double Road, Sadashiv Nagar, Belagavi, Karnataka 590001"),
    ("HCG Cancer Centre — Hubballi", "Gokul Road, Hubballi, Karnataka 580030"),
    ("KIMS Hospital — Oncology, Hubballi", "P.B. Road, Vidyanagar, Hubballi, Karnataka 580021"),
    ("SDM College of Medical Sciences Hospital — Oncology, Dharwad", "Manjushree Nagar, Sattur, Dharwad, Karnataka 580009"),
)

DEMO_DOCTOR_HOSPITALS = {
    "doctor@smartskin.demo": "Kidwai Memorial Institute of Oncology — Bengaluru",
    "doctor.riverside@smartskin.demo": "HCG Cancer Centre — Kalinga Rao Road, Bengaluru",
}

DEMO_DOCTOR_PHONES = {
    "doctor@smartskin.demo": "9876500001",
    "doctor.riverside@smartskin.demo": "9876500002",
}

DEMO_HOSPITAL_ADMINS = {
    "admin.kidwai@smartskin.demo": "Kidwai Memorial Institute of Oncology — Bengaluru",
    "admin.hcg@smartskin.demo": "HCG Cancer Centre — Kalinga Rao Road, Bengaluru",
    "admin.apollo@smartskin.demo": "Apollo Cancer Centre — Bannerghatta Road, Bengaluru",
    "admin.aster@smartskin.demo": "Aster International Institute of Oncology — CMI Hebbal, Bengaluru",
}

# What the doctor studied — shown on profile and to patients.
DOCTOR_QUALIFICATIONS = (
    "MBBS, MD (Dermatology, Venereology & Leprosy)",
    "MBBS, DVD / DDVL (Dermatology)",
    "MBBS, DNB (Dermatology)",
    "MBBS, MD (Dermatology), Fellowship in Dermato-oncology",
    "MBBS, MD (Dermatology), Fellowship in Dermatosurgery",
    "MBBS, MS (General Surgery)",
    "MBBS, MS (General Surgery), MCh (Surgical Oncology)",
    "MBBS, MS (General Surgery), MCh (Plastic Surgery)",
    "MBBS, DNB (Surgical Oncology)",
    "MBBS, MD (General Medicine), DM (Medical Oncology)",
    "MBBS, DNB (Medical Oncology)",
    "MBBS, MD (Radiotherapy / Radiation Oncology)",
    "MBBS, DNB (Radiation Oncology)",
)

DEFAULT_DOCTOR_QUALIFICATION = DOCTOR_QUALIFICATIONS[0]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_phone_digits(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def normalize_hospital_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


class ClinicRepo:
    """Clinic records stored in the same SQLite file as predictions."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def initialize(self) -> None:
        with self.db._connect() as conn:
            conn.executescript(CLINIC_SCHEMA)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
            for col, sql in PRED_EXTRA_COLS.items():
                if col not in cols:
                    conn.execute(sql)
            pcols = {r[1] for r in conn.execute("PRAGMA table_info(patient_profiles)").fetchall()}
            if "date_of_birth" not in pcols:
                conn.execute("ALTER TABLE patient_profiles ADD COLUMN date_of_birth TEXT")
            if "chart_id" not in pcols:
                conn.execute("ALTER TABLE patient_profiles ADD COLUMN chart_id TEXT")
            rcols = {r[1] for r in conn.execute("PRAGMA table_info(clinical_reviews)").fetchall()}
            if "follow_up_notes" not in rcols:
                conn.execute("ALTER TABLE clinical_reviews ADD COLUMN follow_up_notes TEXT")
            dcols = {r[1] for r in conn.execute("PRAGMA table_info(doctor_profiles)").fetchall()}
            if "medical_registration" not in dcols:
                conn.execute(
                    "ALTER TABLE doctor_profiles ADD COLUMN medical_registration TEXT"
                )
            if "hospital_id" not in dcols:
                conn.execute("ALTER TABLE doctor_profiles ADD COLUMN hospital_id INTEGER")
            if "experience" not in dcols:
                conn.execute("ALTER TABLE doctor_profiles ADD COLUMN experience TEXT")
            if "hours_from" not in dcols:
                conn.execute("ALTER TABLE doctor_profiles ADD COLUMN hours_from TEXT")
            if "hours_until" not in dcols:
                conn.execute("ALTER TABLE doctor_profiles ADD COLUMN hours_until TEXT")
            hcols = {r[1] for r in conn.execute("PRAGMA table_info(hospitals)").fetchall()}
            if "address" not in hcols:
                conn.execute("ALTER TABLE hospitals ADD COLUMN address TEXT")
        self._ensure_unique_appointment_slots()
        self.seed_known_hospitals()
        self._prune_non_karnataka_hospitals()
        self._backfill_hospitals()
        self._retire_hospital_admins()
        self._align_demo_doctor_hospitals()

    def _ensure_unique_appointment_slots(self) -> None:
        try:
            with self.db._connect() as conn:
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_appt_unique_slot
                    ON appointments(doctor_id, appt_date, appt_time)
                    WHERE status IN ('requested', 'confirmed', 'rescheduled')
                    """
                )
        except sqlite3.IntegrityError:
            logger.warning("Could not add unique appointment-slot index; duplicate slots exist.")

    def seed_demo_accounts(self) -> None:
        """Create labelled demo logins if the users table is empty of these emails."""
        demos = [
            {
                "email": "patient@smartskin.demo",
                "password": "Patient@123",
                "role": "patient",
                "full_name": "Demo Patient",
                "verified": 1,
                "age": 42,
                "sex": "Female",
            },
            {
                "email": "doctor@smartskin.demo",
                "password": "Doctor@123",
                "role": "doctor",
                "full_name": "Dr. Asha Menon",
                "phone": DEMO_DOCTOR_PHONES["doctor@smartskin.demo"],
                "verified": 1,
                "specialization": DEFAULT_DOCTOR_QUALIFICATION,
                "hospital": "Kidwai Memorial Institute of Oncology — Bengaluru",
                "availability": "Mon–Fri, 09:00–16:00",
                "hours_from": "09:00",
                "hours_until": "16:00",
                "medical_registration": "DEMO-REG-001",
                "experience": "12 years in dermato-oncology",
            },
            {
                "email": "doctor.riverside@smartskin.demo",
                "password": "Doctor@123",
                "role": "doctor",
                "full_name": "Dr. Rohan Iyer",
                "phone": DEMO_DOCTOR_PHONES["doctor.riverside@smartskin.demo"],
                "verified": 1,
                "specialization": DEFAULT_DOCTOR_QUALIFICATION,
                "hospital": "HCG Cancer Centre — Kalinga Rao Road, Bengaluru",
                "availability": "Tue–Sat, 10:00–17:00",
                "hours_from": "10:00",
                "hours_until": "17:00",
                "medical_registration": "DEMO-REG-002",
                "experience": "9 years in dermatology",
            },
            {
                "email": "admin@smartskin.demo",
                "password": "Admin@123",
                "role": "admin",
                "full_name": "System Admin",
                "verified": 1,
                "is_system": 1,
            },
        ]
        for item in demos:
            if self.get_user_by_email(item["email"]):
                continue
            uid = self.create_user(
                email=item["email"],
                password=item["password"],
                role=item["role"],
                full_name=item["full_name"],
                phone=item.get("phone") or "",
                is_verified=item["verified"],
            )
            if item["role"] == "patient":
                self.upsert_patient_profile(
                    uid, age=item.get("age"), sex=item.get("sex")
                )
            if item["role"] == "doctor":
                self.upsert_doctor_profile(
                    uid,
                    specialization=item.get("specialization"),
                    hospital=item.get("hospital"),
                    availability=item.get("availability"),
                    hours_from=item.get("hours_from"),
                    hours_until=item.get("hours_until"),
                    medical_registration=item.get("medical_registration"),
                    experience=item.get("experience"),
                )
            if item["role"] == "admin":
                self.upsert_admin_profile(
                    uid,
                    hospital=item.get("hospital"),
                    is_system=int(item.get("is_system") or 0),
                )
            logger.info("Seeded DEMO account %s", item["email"])

        doctor = self.get_user_by_email("doctor@smartskin.demo")
        patient = self.get_user_by_email("patient@smartskin.demo")
        if doctor and patient:
            self.upsert_patient_profile(
                int(patient["id"]), assigned_doctor_id=int(doctor["id"])
            )

    def _align_demo_doctor_hospitals(self) -> None:
        demo_experience = {
            "doctor@smartskin.demo": "12 years in dermato-oncology",
            "doctor.riverside@smartskin.demo": "9 years in dermatology",
        }
        for email, hospital in DEMO_DOCTOR_HOSPITALS.items():
            user = self.get_user_by_email(email)
            if user and user.get("role") == "doctor":
                extra = {}
                if not ((user.get("profile") or {}).get("experience")):
                    extra["experience"] = demo_experience.get(email, "")
                self.upsert_doctor_profile(int(user["id"]), hospital=hospital, **extra)
                demo_phone = DEMO_DOCTOR_PHONES.get(email) or ""
                if demo_phone and not (user.get("phone") or "").strip():
                    self.update_user_profile(int(user["id"]), user.get("full_name") or "", demo_phone)

    def _align_demo_admins(self) -> None:
        system = self.get_user_by_email("admin@smartskin.demo")
        if system and system.get("role") == "admin":
            self.upsert_admin_profile(int(system["id"]), is_system=1)
        self._retire_hospital_admins()

    def _retire_hospital_admins(self) -> None:
        """Keep a single system admin. Disable hospital-admin demo accounts."""
        with self.db._connect() as conn:
            for email in DEMO_HOSPITAL_ADMINS:
                conn.execute(
                    "UPDATE users SET is_active = 0 WHERE email = ? AND role = 'admin'",
                    (email,),
                )
            conn.execute(
                """
                UPDATE users SET is_active = 0
                WHERE role = 'admin'
                  AND lower(email) != 'admin@smartskin.demo'
                  AND id IN (
                    SELECT user_id FROM admin_profiles WHERE IFNULL(is_system, 0) = 0
                  )
                """
            )
        system = self.get_user_by_email("admin@smartskin.demo")
        if system and system.get("role") == "admin":
            self.upsert_admin_profile(int(system["id"]), is_system=1)

    # ----- users -----
    def create_user(
        self,
        email: str,
        password: str,
        role: str,
        full_name: str,
        phone: str = "",
        is_verified: int = 0,
    ) -> int:
        email = email.strip().lower()
        with self.db._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (email, password_hash, role, full_name, phone,
                                   is_active, is_verified, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    email,
                    generate_password_hash(password),
                    role,
                    full_name.strip(),
                    phone.strip() if phone else None,
                    int(is_verified),
                    _now(),
                ),
            )
            return int(cur.lastrowid)

    def authenticate(
        self, email: str, password: str, hospital: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_email(email)
        if not user or not user.get("is_active"):
            return None
        if not check_password_hash(user["password_hash"], password):
            return None
        if user["role"] == "admin":
            if not self.is_system_admin(user):
                return None
        return user

    def authenticate_with_password(
        self, identifier: str, password: str, role: str
    ) -> Optional[Dict[str, Any]]:
        ident = (identifier or "").strip()
        if not ident or not password:
            return None
        if "@" in ident:
            user = self.get_user_by_email(ident.lower())
        else:
            user = self.get_user_by_phone(ident, role=role)
        if not user or user.get("role") != role or not user.get("is_active"):
            return None
        if not check_password_hash(user["password_hash"], password):
            return None
        if role == "admin" and not self.is_system_admin(user):
            return None
        return user

    def hospital_matches(self, user: Dict[str, Any], hospital_name: str) -> bool:
        key = normalize_hospital_name(hospital_name)
        if not key:
            return False
        profile = user.get("profile") or {}
        stored = normalize_hospital_name(str(profile.get("hospital") or ""))
        if stored and stored == key:
            return True
        hid = profile.get("hospital_id")
        if hid:
            hospital = self.get_hospital(int(hid))
            if hospital and hospital["name_key"] == key:
                return True
        return False

    def get_hospital(self, hospital_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hospitals WHERE id = ?", (hospital_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_hospitals(self) -> List[Dict[str, Any]]:
        allowed = {normalize_hospital_name(name) for name, _ in KNOWN_CANCER_HOSPITALS}
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hospitals ORDER BY name COLLATE NOCASE"
            ).fetchall()
        items = [dict(r) for r in rows if r["name_key"] in allowed]

        def _sort_key(hospital: Dict[str, Any]) -> tuple:
            blob = f"{hospital.get('name') or ''} {hospital.get('address') or ''}".lower()
            in_bengaluru = "bengaluru" in blob or "bangalore" in blob
            return (0 if in_bengaluru else 1, hospital["name"].lower())

        return sorted(items, key=_sort_key)

    def list_qualifications(self, current: str = "") -> List[str]:
        items = list(DOCTOR_QUALIFICATIONS)
        extra = (current or "").strip()
        if extra and extra not in items:
            items = [extra] + items
        return items

    def is_allowed_qualification(self, value: str, current: str = "") -> bool:
        chosen = (value or "").strip()
        if chosen in DOCTOR_QUALIFICATIONS:
            return True
        return bool(current) and chosen == current.strip()

    def seed_known_hospitals(self) -> None:
        for name, address in KNOWN_CANCER_HOSPITALS:
            self.get_or_create_hospital(name, address=address, replace_address=True)

    def _prune_non_karnataka_hospitals(self) -> None:
        allowed = {normalize_hospital_name(name) for name, _ in KNOWN_CANCER_HOSPITALS}
        with self.db._connect() as conn:
            used = {
                int(r[0])
                for r in conn.execute(
                    "SELECT hospital_id FROM doctor_profiles WHERE hospital_id IS NOT NULL"
                ).fetchall()
            }
            rows = conn.execute("SELECT id, name_key FROM hospitals").fetchall()
            for row in rows:
                if row["name_key"] in allowed:
                    continue
                if int(row["id"]) in used:
                    continue
                conn.execute("DELETE FROM hospitals WHERE id = ?", (row["id"],))

    def get_or_create_hospital(
        self, name: str, address: str = "", replace_address: bool = False
    ) -> Optional[Dict[str, Any]]:
        display = " ".join((name or "").strip().split())
        key = normalize_hospital_name(display)
        addr = " ".join((address or "").strip().split())
        if not key:
            return None
        allowed = {normalize_hospital_name(item[0]) for item in KNOWN_CANCER_HOSPITALS}
        if key not in allowed:
            return None
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hospitals WHERE name_key = ?", (key,)
            ).fetchone()
            if row:
                current = dict(row)
                if addr and (replace_address or not (current.get("address") or "").strip()):
                    conn.execute(
                        "UPDATE hospitals SET address = ? WHERE id = ?",
                        (addr, current["id"]),
                    )
                    current["address"] = addr
                return current
            cur = conn.execute(
                """
                INSERT INTO hospitals (name, name_key, address, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (display, key, addr or None, _now()),
            )
            return {
                "id": int(cur.lastrowid),
                "name": display,
                "name_key": key,
                "address": addr or None,
                "created_at": _now(),
            }

    def _backfill_hospitals(self) -> None:
        with self.db._connect() as conn:
            rows = conn.execute(
                "SELECT user_id, hospital, hospital_id FROM doctor_profiles"
            ).fetchall()
        for row in rows:
            name = (row["hospital"] or "").strip()
            if not name:
                continue
            hospital = self.get_or_create_hospital(name)
            if not hospital:
                continue
            if row["hospital_id"] == hospital["id"] and row["hospital"] == hospital["name"]:
                continue
            with self.db._connect() as conn:
                conn.execute(
                    """
                    UPDATE doctor_profiles
                    SET hospital = ?, hospital_id = ?
                    WHERE user_id = ?
                    """,
                    (hospital["name"], hospital["id"], row["user_id"]),
                )

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._hydrate_user(dict(row)) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return self._hydrate_user(dict(row)) if row else None

    def get_user_by_phone(
        self, phone: str, role: Optional[str] = "patient"
    ) -> Optional[Dict[str, Any]]:
        digits = normalize_phone_digits(phone)
        if len(digits) < 10:
            return None
        key = digits[-10:]
        sql = "SELECT * FROM users WHERE phone IS NOT NULL AND phone != ''"
        args: List[Any] = []
        if role:
            sql += " AND role = ?"
            args.append(role)
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        for row in rows:
            stored = normalize_phone_digits(row["phone"] or "")
            if len(stored) >= 10 and stored[-10:] == key:
                return self._hydrate_user(dict(row))
        return None

    def authenticate_patient_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_phone(phone)
        if not user or not user.get("is_active"):
            return None
        return user

    def _hydrate_user(self, user: Dict[str, Any]) -> Dict[str, Any]:
        uid = int(user["id"])
        if user["role"] == "patient":
            user["profile"] = self.get_patient_profile(uid) or {}
        elif user["role"] == "doctor":
            user["profile"] = self.get_doctor_profile(uid) or {}
        elif user["role"] == "admin":
            user["profile"] = self.get_admin_profile(uid) or {}
        else:
            user["profile"] = {}
        return user

    def is_system_admin(self, user: Optional[Dict[str, Any]]) -> bool:
        if not user or user.get("role") != "admin":
            return False
        return int((user.get("profile") or {}).get("is_system") or 0) == 1

    def hospital_scope_id(self, user: Optional[Dict[str, Any]]) -> Optional[int]:
        if not user:
            return None
        hid = (user.get("profile") or {}).get("hospital_id")
        try:
            return int(hid) if hid is not None else None
        except (TypeError, ValueError):
            return None

    def user_in_hospital_scope(
        self, admin: Dict[str, Any], target: Optional[Dict[str, Any]]
    ) -> bool:
        if self.is_system_admin(admin):
            return True
        hid = self.hospital_scope_id(admin)
        if not hid or not target:
            return False
        th = (target.get("profile") or {}).get("hospital_id")
        try:
            return th is not None and int(th) == hid
        except (TypeError, ValueError):
            return False

    def list_users(
        self, role: Optional[str] = None, hospital_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM users"
        args: tuple = ()
        if role:
            sql += " WHERE role = ?"
            args = (role,)
        sql += " ORDER BY datetime(created_at) DESC"
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        users = [self._hydrate_user(dict(r)) for r in rows]
        if hospital_id is None:
            return users
        hid = int(hospital_id)
        scoped = []
        for item in users:
            if item["role"] not in {"doctor", "admin"}:
                continue
            if self.is_system_admin(item):
                continue
            ph = (item.get("profile") or {}).get("hospital_id")
            try:
                if ph is not None and int(ph) == hid:
                    scoped.append(item)
            except (TypeError, ValueError):
                continue
        return scoped

    def list_verified_doctors(self) -> List[Dict[str, Any]]:
        with self.db._connect() as conn:
            rows = conn.execute(
                """
                SELECT u.*, d.specialization, d.hospital, d.hospital_id, d.availability,
                       d.hours_from, d.hours_until, d.medical_registration, d.experience
                FROM users u
                LEFT JOIN doctor_profiles d ON d.user_id = u.id
                WHERE u.role = 'doctor' AND u.is_active = 1 AND u.is_verified = 1
                ORDER BY u.full_name
                """
            ).fetchall()
        return [self._public_doctor_card(dict(r)) for r in rows]

    def _public_doctor_card(self, doctor: Dict[str, Any]) -> Dict[str, Any]:
        spec = (doctor.get("specialization") or "").strip()
        exp = (doctor.get("experience") or "").strip()
        hospital = (doctor.get("hospital") or "").strip()
        bits = []
        if spec:
            bits.append(spec + ".")
        if exp:
            bits.append(exp + ("." if not exp.endswith(".") else ""))
        if hospital:
            bits.append(f"Practises at {hospital}.")
        start, end = parse_availability_hours(
            str(doctor.get("availability") or ""),
            str(doctor.get("hours_from") or ""),
            str(doctor.get("hours_until") or ""),
        )
        doctor["hour_slots"] = hour_slots_for_range(start, end)
        bits.append(f"Available {_clock_hour(start)}–{_clock_hour(end)}.")
        doctor["description"] = " ".join(bits) or (
            "Dermatology clinician available for in-person consultation."
        )
        return doctor

    def list_hospitals_for_booking(self, query: str = "") -> List[Dict[str, Any]]:
        needle = (query or "").strip().lower()
        doctors = self.list_verified_doctors()
        counts: Dict[Any, int] = {}
        for doc in doctors:
            hid = doc.get("hospital_id")
            key = int(hid) if hid not in (None, "") else (doc.get("hospital") or "")
            if key == "":
                continue
            counts[key] = counts.get(key, 0) + 1
        out: List[Dict[str, Any]] = []
        for hospital in self.list_hospitals():
            n = counts.get(int(hospital["id"]), 0) or counts.get(hospital.get("name") or "", 0)
            if n <= 0:
                continue
            blob = f"{hospital.get('name') or ''} {hospital.get('address') or ''}".lower()
            if needle and needle not in blob:
                continue
            item = dict(hospital)
            item["doctor_count"] = n
            out.append(item)
        return out

    def list_doctors_at_hospital(self, hospital_id: int) -> List[Dict[str, Any]]:
        hospital = self.get_hospital(int(hospital_id))
        if not hospital:
            return []
        name_key = hospital.get("name_key") or ""
        matched = []
        for doc in self.list_verified_doctors():
            hid = doc.get("hospital_id")
            if hid not in (None, "") and int(hid) == int(hospital["id"]):
                matched.append(doc)
                continue
            if name_key and normalize_hospital_name(str(doc.get("hospital") or "")) == name_key:
                matched.append(doc)
        return matched

    def set_user_flags(
        self, user_id: int, is_active: Optional[int] = None, is_verified: Optional[int] = None
    ) -> None:
        updates = []
        vals = []
        if is_active is not None:
            updates.append("is_active = ?")
            vals.append(int(is_active))
        if is_verified is not None:
            updates.append("is_verified = ?")
            vals.append(int(is_verified))
        if not updates:
            return
        vals.append(user_id)
        with self.db._connect() as conn:
            conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", vals)

    def delete_account(self, user_id: int, role: str) -> bool:
        """Remove a doctor or patient from active use. Reports stay on file."""
        if role not in {"doctor", "patient"}:
            return False
        user = self.get_user(user_id)
        if not user or user.get("role") != role or not user.get("is_active"):
            return False
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        retired_email = f"deleted.{user_id}.{stamp}@deleted.local"
        with self.db._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET is_active = 0, email = ?, phone = NULL
                WHERE id = ? AND role = ?
                """,
                (retired_email, user_id, role),
            )
        return True

    def upsert_patient_profile(self, user_id: int, **fields: Any) -> None:
        existing = self.get_patient_profile(user_id)
        allowed = {"age", "sex", "address", "assigned_doctor_id", "date_of_birth", "chart_id"}
        data = {k: fields[k] for k in allowed if k in fields}
        with self.db._connect() as conn:
            if not existing:
                conn.execute(
                    """
                    INSERT INTO patient_profiles
                        (user_id, age, sex, address, assigned_doctor_id, date_of_birth, chart_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        data.get("age"),
                        data.get("sex"),
                        data.get("address"),
                        data.get("assigned_doctor_id"),
                        data.get("date_of_birth"),
                        data.get("chart_id"),
                    ),
                )
            elif data:
                sets = ", ".join(f"{k} = ?" for k in data)
                conn.execute(
                    f"UPDATE patient_profiles SET {sets} WHERE user_id = ?",
                    list(data.values()) + [user_id],
                )

    def get_patient_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patient_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_doctor_profile(self, user_id: int, **fields: Any) -> None:
        existing = self.get_doctor_profile(user_id)
        allowed = {
            "specialization",
            "hospital",
            "hospital_id",
            "availability",
            "hours_from",
            "hours_until",
            "medical_registration",
            "experience",
        }
        data = {k: fields[k] for k in allowed if k in fields}
        if "hospital" in data:
            hospital = self.get_or_create_hospital(str(data.get("hospital") or ""))
            if hospital:
                data["hospital"] = hospital["name"]
                data["hospital_id"] = hospital["id"]
            else:
                data["hospital"] = None
                data["hospital_id"] = None
        with self.db._connect() as conn:
            if not existing:
                conn.execute(
                    """
                    INSERT INTO doctor_profiles
                        (user_id, specialization, hospital, hospital_id, availability, hours_from, hours_until, medical_registration, experience)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        data.get("specialization"),
                        data.get("hospital"),
                        data.get("hospital_id"),
                        data.get("availability"),
                        data.get("hours_from"),
                        data.get("hours_until"),
                        data.get("medical_registration"),
                        data.get("experience"),
                    ),
                )
            elif data:
                sets = ", ".join(f"{k} = ?" for k in data)
                conn.execute(
                    f"UPDATE doctor_profiles SET {sets} WHERE user_id = ?",
                    list(data.values()) + [user_id],
                )

    def get_doctor_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM doctor_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_admin_profile(
        self, user_id: int, hospital: str = "", is_system: Optional[int] = None
    ) -> None:
        existing = self.get_admin_profile(user_id) or {}
        sys_flag = int(is_system) if is_system is not None else int(existing.get("is_system") or 0)
        hosp_name = existing.get("hospital")
        hosp_id = existing.get("hospital_id")
        if sys_flag:
            hosp_name = None
            hosp_id = None
        elif hospital:
            found = self.get_or_create_hospital(hospital)
            if found:
                hosp_name = found["name"]
                hosp_id = found["id"]
        with self.db._connect() as conn:
            if not existing:
                conn.execute(
                    """
                    INSERT INTO admin_profiles (user_id, hospital, hospital_id, is_system)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, hosp_name, hosp_id, sys_flag),
                )
            else:
                conn.execute(
                    """
                    UPDATE admin_profiles
                    SET hospital = ?, hospital_id = ?, is_system = ?
                    WHERE user_id = ?
                    """,
                    (hosp_name, hosp_id, sys_flag, user_id),
                )

    def get_admin_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM admin_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_user_profile(self, user_id: int, full_name: str, phone: str = "") -> None:
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE users SET full_name = ?, phone = ? WHERE id = ?",
                (full_name.strip(), phone.strip() or None, user_id),
            )

    # ----- cases / predictions -----
    def attach_case_owner(
        self,
        pred_id: int,
        owner_user_id: Optional[int],
        assigned_doctor_id: Optional[int],
        review_status: str = "pending_review",
        report_code: Optional[str] = None,
    ) -> None:
        with self.db._connect() as conn:
            conn.execute(
                """
                UPDATE predictions
                SET owner_user_id = ?, assigned_doctor_id = ?,
                    review_status = COALESCE(review_status, ?),
                    report_code = COALESCE(report_code, ?)
                WHERE id = ?
                """,
                (owner_user_id, assigned_doctor_id, review_status, report_code, pred_id),
            )

    def set_case_review_status(self, pred_id: int, status: str) -> None:
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE predictions SET review_status = ? WHERE id = ?",
                (status, pred_id),
            )

    def patient_cases(self, user_id: int, finalized_only: bool = False) -> List[Dict[str, Any]]:
        sql = """
            SELECT p.*, u.full_name AS doctor_name, cr.final_diagnosis
            FROM predictions p
            LEFT JOIN users u ON u.id = p.assigned_doctor_id
            LEFT JOIN clinical_reviews cr ON cr.case_id = p.id
            WHERE p.owner_user_id = ?
        """
        if finalized_only:
            sql += " AND p.review_status = 'finalized'"
        sql += " ORDER BY datetime(p.created_at) DESC"
        with self.db._connect() as conn:
            rows = conn.execute(sql, (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def doctor_cases(
        self, doctor_id: int, status: Optional[str] = None, search: str = ""
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT p.*, u.full_name AS patient_account_name, u.email AS patient_email
            FROM predictions p
            LEFT JOIN users u ON u.id = p.owner_user_id
            WHERE (p.assigned_doctor_id = ? OR p.assigned_doctor_id IS NULL)
        """
        args: List[Any] = [doctor_id]
        if status:
            sql += " AND COALESCE(p.review_status, 'pending_review') = ?"
            args.append(status)
        if search.strip():
            sql += """ AND (
                p.patient_name LIKE ? OR p.patient_id LIKE ?
                OR CAST(p.id AS TEXT) LIKE ? OR COALESCE(u.phone, '') LIKE ?
            )"""
            q = f"%{search.strip()}%"
            args.extend([q, q, q, q])
        sql += " ORDER BY datetime(p.created_at) DESC"
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def doctor_stats(self, doctor_id: int) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        with self.db._connect() as conn:
            today_analyses = conn.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE assigned_doctor_id = ? AND date(created_at) = ?
                """,
                (doctor_id, today),
            ).fetchone()[0]
            completed = conn.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE assigned_doctor_id = ?
                """,
                (doctor_id,),
            ).fetchone()[0]
            melanoma = conn.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE assigned_doctor_id = ? AND prediction = 'Malignant'
                """,
                (doctor_id,),
            ).fetchone()[0]
            reports = conn.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE assigned_doctor_id = ? AND review_status = 'finalized'
                """,
                (doctor_id,),
            ).fetchone()[0]
            patients = conn.execute(
                """
                SELECT COUNT(DISTINCT owner_user_id) FROM predictions
                WHERE assigned_doctor_id = ? AND owner_user_id IS NOT NULL
                """,
                (doctor_id,),
            ).fetchone()[0]
        return {
            "today_analyses": today_analyses,
            "analyses_completed": completed,
            "melanoma_predictions": melanoma,
            "reports_generated": reports,
            "total_patients": patients,
        }

    def can_access_case(self, user: Dict[str, Any], record: Dict[str, Any]) -> bool:
        if not user or not record:
            return False
        role = user.get("role")
        uid = int(user["id"])
        if role == "admin":
            if self.is_system_admin(user):
                return True
            hid = self.hospital_scope_id(user)
            if not hid:
                return False
            assigned = record.get("assigned_doctor_id")
            if not assigned:
                return False
            doctor = self.get_user(int(assigned))
            dh = (doctor or {}).get("profile") or {}
            try:
                return int(dh.get("hospital_id")) == hid
            except (TypeError, ValueError):
                return False
        if role == "doctor" and user.get("is_verified"):
            assigned = record.get("assigned_doctor_id")
            return assigned in (None, uid) or int(assigned) == uid
        if role == "patient":
            return record.get("owner_user_id") == uid
        return False

    def list_doctor_patients(self, doctor_id: int, search: str = "") -> List[Dict[str, Any]]:
        sql = """
            SELECT u.id, u.full_name, u.email, u.phone, pp.age, pp.sex, pp.chart_id,
                   COUNT(p.id) AS case_count,
                   MAX(p.created_at) AS last_case
            FROM users u
            JOIN patient_profiles pp ON pp.user_id = u.id
            LEFT JOIN predictions p ON p.owner_user_id = u.id
            WHERE u.role = 'patient' AND (
                pp.assigned_doctor_id = ?
                OR p.assigned_doctor_id = ?
            )
        """
        args: List[Any] = [doctor_id, doctor_id]
        if search.strip():
            sql += """ AND (
                u.full_name LIKE ? OR u.phone LIKE ? OR CAST(u.id AS TEXT) LIKE ?
                OR COALESCE(pp.chart_id, '') LIKE ?
            )"""
            q = f"%{search.strip()}%"
            args.extend([q, q, q, q])
        sql += " GROUP BY u.id ORDER BY u.full_name"
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    # ----- clinical review -----
    def get_review(self, case_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM clinical_reviews WHERE case_id = ?", (case_id,)
            ).fetchone()
        return dict(row) if row else None

    def save_review(self, case_id: int, doctor_id: int, payload: Dict[str, Any]) -> None:
        existing = self.get_review(case_id)
        fields = [
            "lesion_location",
            "lesion_duration",
            "symptoms",
            "clinical_observations",
            "doctor_notes",
            "ai_agreement",
            "final_diagnosis",
            "recommendations",
            "follow_up_notes",
            "review_status",
        ]
        data = {k: payload.get(k) for k in fields}
        data["doctor_id"] = doctor_id
        if payload.get("review_status") == "finalized":
            data["verified_at"] = _now()
        with self.db._connect() as conn:
            if existing:
                sets = ", ".join(f"{k} = ?" for k in data)
                conn.execute(
                    f"UPDATE clinical_reviews SET {sets} WHERE case_id = ?",
                    list(data.values()) + [case_id],
                )
            else:
                data["case_id"] = case_id
                cols = ", ".join(data.keys())
                qs = ", ".join(["?"] * len(data))
                conn.execute(
                    f"INSERT INTO clinical_reviews ({cols}) VALUES ({qs})",
                    list(data.values()),
                )
        self.set_case_review_status(case_id, data.get("review_status") or "draft")

    # ----- appointments -----
    def book_appointment(
        self, patient_id: int, doctor_id: int, appt_date: str, appt_time: str, reason: str
    ) -> int:
        with self.db._connect() as conn:
            clash = conn.execute(
                """
                SELECT id FROM appointments
                WHERE doctor_id = ? AND appt_date = ? AND appt_time = ?
                  AND status IN ('requested', 'confirmed', 'rescheduled')
                """,
                (doctor_id, appt_date, appt_time),
            ).fetchone()
            if clash:
                raise ValueError("That time slot is not available.")
            try:
                cur = conn.execute(
                    """
                    INSERT INTO appointments
                        (patient_id, doctor_id, appt_date, appt_time, reason, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'requested', ?)
                    """,
                    (patient_id, doctor_id, appt_date, appt_time, reason.strip(), _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("That time slot is not available.") from exc
            return int(cur.lastrowid)

    def list_appointments(
        self, *, patient_id: Optional[int] = None, doctor_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT a.*,
                   p.full_name AS patient_name,
                   p.phone AS patient_phone,
                   d.full_name AS doctor_name,
                   dp.hospital AS doctor_hospital
            FROM appointments a
            JOIN users p ON p.id = a.patient_id
            JOIN users d ON d.id = a.doctor_id
            LEFT JOIN doctor_profiles dp ON dp.user_id = d.id
            WHERE 1=1
        """
        args: List[Any] = []
        if patient_id:
            sql += " AND a.patient_id = ?"
            args.append(patient_id)
        if doctor_id:
            sql += " AND a.doctor_id = ?"
            args.append(doctor_id)
        sql += " ORDER BY a.appt_date DESC, a.appt_time DESC"
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["time_label"] = slot_label(str(item.get("appt_time") or ""))
            out.append(item)
        return out

    def get_appointment(self, appt_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                """
                SELECT a.*, p.full_name AS patient_name, p.phone AS patient_phone,
                       d.full_name AS doctor_name
                FROM appointments a
                JOIN users p ON p.id = a.patient_id
                JOIN users d ON d.id = a.doctor_id
                WHERE a.id = ?
                """,
                (appt_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["time_label"] = slot_label(str(item.get("appt_time") or ""))
        return item

    def set_appointment_status(self, appt_id: int, status: str) -> None:
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE appointments SET status = ? WHERE id = ?", (status, appt_id)
            )

    def reschedule_appointment(
        self, appt_id: int, appt_date: str, appt_time: str
    ) -> None:
        current = self.get_appointment(appt_id)
        if not current:
            raise ValueError("Appointment not found.")
        doctor = self.get_user(int(current["doctor_id"]))
        if appt_time not in {s["value"] for s in doctor_hour_slots(doctor)}:
            raise ValueError("Select a valid time slot.")
        with self.db._connect() as conn:
            clash = conn.execute(
                """
                SELECT id FROM appointments
                WHERE doctor_id = ? AND appt_date = ? AND appt_time = ?
                  AND status IN ('requested', 'confirmed', 'rescheduled')
                  AND id != ?
                """,
                (current["doctor_id"], appt_date, appt_time, appt_id),
            ).fetchone()
            if clash:
                raise ValueError("That time slot is not available.")
            try:
                conn.execute(
                    """
                    UPDATE appointments
                    SET appt_date = ?, appt_time = ?, status = 'rescheduled'
                    WHERE id = ?
                    """,
                    (appt_date, appt_time, appt_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("That time slot is not available.") from exc

    def taken_slots(self, doctor_id: int, appt_date: str, exclude_id: Optional[int] = None) -> List[str]:
        sql = """
            SELECT appt_time FROM appointments
            WHERE doctor_id = ? AND appt_date = ?
              AND status IN ('requested', 'confirmed', 'rescheduled')
        """
        args: List[Any] = [int(doctor_id), appt_date]
        if exclude_id:
            sql += " AND id != ?"
            args.append(int(exclude_id))
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [str(r["appt_time"]) for r in rows]

    # ----- prescriptions -----
    def create_prescription(
        self,
        patient_id: int,
        doctor_id: int,
        diagnosis: str,
        notes: str,
        items: List[Dict[str, str]],
        case_id: Optional[int] = None,
        code: str = "",
        pdf_path: str = "",
    ) -> int:
        with self.db._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO prescriptions
                    (code, patient_id, doctor_id, case_id, diagnosis, notes, pdf_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (code, patient_id, doctor_id, case_id, diagnosis, notes, pdf_path, _now()),
            )
            pid = int(cur.lastrowid)
            for item in items:
                if not (item.get("medicine") or "").strip():
                    continue
                conn.execute(
                    """
                    INSERT INTO prescription_items
                        (prescription_id, medicine, dosage, frequency, duration, instructions)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pid,
                        item.get("medicine", "").strip(),
                        item.get("dosage", "").strip(),
                        item.get("frequency", "").strip(),
                        item.get("duration", "").strip(),
                        item.get("instructions", "").strip(),
                    ),
                )
        return pid

    def set_prescription_pdf(self, prescription_id: int, pdf_path: str, code: str) -> None:
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE prescriptions SET pdf_path = ?, code = ? WHERE id = ?",
                (pdf_path, code, prescription_id),
            )

    def get_prescription(self, prescription_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                """
                SELECT pr.*, p.full_name AS patient_name, d.full_name AS doctor_name
                FROM prescriptions pr
                JOIN users p ON p.id = pr.patient_id
                JOIN users d ON d.id = pr.doctor_id
                WHERE pr.id = ?
                """,
                (prescription_id,),
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            items = conn.execute(
                "SELECT * FROM prescription_items WHERE prescription_id = ?",
                (prescription_id,),
            ).fetchall()
            data["items"] = [dict(i) for i in items]
        return data

    def get_prescription_by_case(self, case_id: int) -> Optional[Dict[str, Any]]:
        with self.db._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM prescriptions
                WHERE case_id = ?
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT 1
                """,
                (int(case_id),),
            ).fetchone()
        if not row:
            return None
        return self.get_prescription(int(row["id"]))

    def delete_prescription(self, prescription_id: int) -> None:
        with self.db._connect() as conn:
            conn.execute(
                "DELETE FROM prescription_items WHERE prescription_id = ?",
                (prescription_id,),
            )
            conn.execute("DELETE FROM prescriptions WHERE id = ?", (prescription_id,))

    def upsert_case_prescription(
        self,
        case_id: int,
        patient_id: int,
        doctor_id: int,
        diagnosis: str,
        notes: str,
        items: List[Dict[str, str]],
        code: str = "",
    ) -> int:
        existing = self.get_prescription_by_case(case_id)
        if existing:
            self.delete_prescription(int(existing["id"]))
        return self.create_prescription(
            patient_id=patient_id,
            doctor_id=doctor_id,
            diagnosis=diagnosis,
            notes=notes,
            items=items,
            case_id=case_id,
            code=code,
        )

    def list_prescriptions(
        self, *, patient_id: Optional[int] = None, doctor_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT pr.*, p.full_name AS patient_name, d.full_name AS doctor_name
            FROM prescriptions pr
            JOIN users p ON p.id = pr.patient_id
            JOIN users d ON d.id = pr.doctor_id
            WHERE 1=1
        """
        args: List[Any] = []
        if patient_id:
            sql += " AND pr.patient_id = ?"
            args.append(patient_id)
        if doctor_id:
            sql += " AND pr.doctor_id = ?"
            args.append(doctor_id)
        sql += " ORDER BY datetime(pr.created_at) DESC"
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
            out = []
            for r in rows:
                rec = dict(r)
                items = conn.execute(
                    "SELECT * FROM prescription_items WHERE prescription_id = ?",
                    (rec["id"],),
                ).fetchall()
                rec["items"] = [dict(i) for i in items]
                out.append(rec)
        return out

    # ----- notifications / audit -----
    def notify(self, user_id: int, title: str, body: str = "", link: str = "") -> None:
        with self.db._connect() as conn:
            conn.execute(
                """
                INSERT INTO notifications (user_id, title, body, link, is_read, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (user_id, title, body, link, _now()),
            )

    def list_notifications(self, user_id: int, unread_only: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM notifications WHERE user_id = ?"
        if unread_only:
            sql += " AND is_read = 0"
        sql += " ORDER BY datetime(created_at) DESC LIMIT 50"
        with self.db._connect() as conn:
            rows = conn.execute(sql, (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def mark_notifications_read(self, user_id: int) -> None:
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,)
            )

    def audit(
        self,
        user_id: Optional[int],
        action: str,
        entity: str = "",
        entity_id: str = "",
        details: str = "",
    ) -> None:
        with self.db._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (user_id, action, entity, entity_id, details, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, action, entity, str(entity_id), details, _now()),
            )

    def list_audit(
        self, limit: int = 100, hospital_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        sql = """
                SELECT a.*, u.email, u.full_name
                FROM audit_logs a
                LEFT JOIN users u ON u.id = a.user_id
        """
        args: List[Any] = []
        if hospital_id is not None:
            staff = self.list_users(hospital_id=int(hospital_id))
            ids = [int(s["id"]) for s in staff]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            sql += f" WHERE a.user_id IN ({placeholders})"
            args.extend(ids)
        sql += " ORDER BY a.id DESC LIMIT ?"
        args.append(limit)
        with self.db._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def admin_stats(self, hospital_id: Optional[int] = None) -> Dict[str, Any]:
        with self.db._connect() as conn:
            if hospital_id is None:
                users = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE is_active = 1"
                ).fetchone()[0]
                doctors = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'doctor' AND is_active = 1"
                ).fetchone()[0]
                patients = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'patient' AND is_active = 1"
                ).fetchone()[0]
                cases = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                pending_docs = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'doctor' AND is_active = 1 AND is_verified = 0"
                ).fetchone()[0]
            else:
                hid = int(hospital_id)
                doctors = conn.execute(
                    """
                    SELECT COUNT(*) FROM users u
                    JOIN doctor_profiles d ON d.user_id = u.id
                    WHERE u.role = 'doctor' AND d.hospital_id = ?
                    """,
                    (hid,),
                ).fetchone()[0]
                pending_docs = conn.execute(
                    """
                    SELECT COUNT(*) FROM users u
                    JOIN doctor_profiles d ON d.user_id = u.id
                    WHERE u.role = 'doctor' AND u.is_verified = 0 AND d.hospital_id = ?
                    """,
                    (hid,),
                ).fetchone()[0]
                patients = conn.execute(
                    """
                    SELECT COUNT(DISTINCT pp.user_id) FROM patient_profiles pp
                    JOIN doctor_profiles d ON d.user_id = pp.assigned_doctor_id
                    WHERE d.hospital_id = ?
                    """,
                    (hid,),
                ).fetchone()[0]
                cases = conn.execute(
                    """
                    SELECT COUNT(*) FROM predictions p
                    JOIN doctor_profiles d ON d.user_id = p.assigned_doctor_id
                    WHERE d.hospital_id = ?
                    """,
                    (hid,),
                ).fetchone()[0]
                admins = conn.execute(
                    """
                    SELECT COUNT(*) FROM admin_profiles
                    WHERE hospital_id = ? AND is_system = 0
                    """,
                    (hid,),
                ).fetchone()[0]
                users = doctors + admins
        return {
            "users": users,
            "doctors": doctors,
            "patients": patients,
            "cases": cases,
            "pending_doctors": pending_docs,
        }


_clinic: Optional[ClinicRepo] = None


def get_clinic() -> ClinicRepo:
    global _clinic
    if _clinic is None:
        repo = ClinicRepo(get_db())
        repo.initialize()
        _clinic = repo
    return _clinic
