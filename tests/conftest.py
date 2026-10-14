"""Seed demo accounts for tests only — they are not created in the running app."""

from database.clinic import get_clinic


def pytest_configure():
    get_clinic().seed_demo_accounts()
    get_clinic()._align_demo_doctor_hospitals()
    get_clinic()._align_demo_admins()
