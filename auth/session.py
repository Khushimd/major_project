"""Flask session helpers and route guards."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Optional

from flask import flash, g, redirect, request, session, url_for

from database.clinic import get_clinic


def current_user() -> Optional[Dict[str, Any]]:
    if getattr(g, "user", None) is not None:
        return g.user
    uid = session.get("user_id")
    if not uid:
        g.user = None
        return None
    g.user = get_clinic().get_user(int(uid))
    return g.user


def login_user(user: Dict[str, Any]) -> None:
    session.clear()
    session["user_id"] = int(user["id"])
    session["role"] = user["role"]
    session.permanent = True
    g.user = user


def logout_user() -> None:
    session.clear()
    g.user = None


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        if not user.get("is_active"):
            logout_user()
            flash("This account is inactive.", "danger")
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped



def role_required(*roles: str) -> Callable:
    def decorator(view: Callable) -> Callable:
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                flash("Please sign in to continue.", "warning")
                return redirect(url_for("auth.login", next=request.path))
            if user.get("role") not in roles:
                flash("You do not have access to that page.", "danger")
                return redirect(_dashboard_for(user))
            if user.get("role") == "doctor" and not user.get("is_verified"):
                flash("Your doctor account is pending admin verification.", "warning")
                return redirect(url_for("auth.login"))
            return view(*args, **kwargs)

        return wrapped

    return decorator


def _dashboard_for(user: Dict[str, Any]) -> str:
    endpoint = dashboard_endpoint(user)
    return url_for(endpoint or "home")


def dashboard_endpoint(user: Optional[Dict[str, Any]]) -> Optional[str]:
    if not user:
        return None
    role = user.get("role")
    if role == "doctor":
        return "doctor.dashboard"
    if role == "patient":
        return "patient.dashboard"
    if role == "admin":
        return "admin.dashboard"
    return "home"
