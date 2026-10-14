"""
Flask web application — melanoma clinical decision support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from datetime import timedelta
import time
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from auth.session import current_user, dashboard_endpoint, login_required, logout_user, role_required
from blueprints.admin_bp import bp as admin_bp
from blueprints.auth_bp import bp as auth_bp
from blueprints.doctor_bp import bp as doctor_bp
from blueprints.patient_bp import bp as patient_bp
from database.clinic import get_clinic
from database.db import get_db
from models.ensemble import resolve_decision_threshold
from utils.helpers import allowed_file, generate_unique_id, safe_filename
from utils.logger import setup_logger
from utils.pipeline import pipeline
from utils.web import display_model_name, display_prediction, is_melanoma, to_static_url

logger = setup_logger("app")

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)
app.secret_key = config.FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(config.UPLOAD_DIR)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
IDLE_SECONDS = 45 * 60

app.register_blueprint(auth_bp)
app.register_blueprint(patient_bp)
app.register_blueprint(doctor_bp)
app.register_blueprint(admin_bp)


@app.before_request
def _secure_session():
    get_clinic()
    if request.endpoint == "static":
        return None
    uid = session.get("user_id")
    if uid:
        last = session.get("last_activity")
        now = time.time()
        if last and (now - float(last)) > IDLE_SECONDS:
            logout_user()
            flash("Session expired. Please sign in again.", "warning")
            return redirect(url_for("auth.login"))
        session["last_activity"] = now
    g.user = current_user()
    return None


@app.context_processor
def inject_globals():
    user = getattr(g, "user", None)
    dest = dashboard_endpoint(user) if user else None
    return {
        "disclaimer": config.MEDICAL_DISCLAIMER,
        "app_title": "Melanoma CDSS",
        "full_title": "Melanoma clinical decision support",
        "current_user": user,
        "dashboard_url": url_for(dest) if dest else url_for("home"),
        "ai_assist_note": (
            "This result is intended to support clinical assessment "
            "and does not replace professional diagnosis."
        ),
    }


@app.template_filter("ai_label")
def _ai_label_filter(value):
    return display_prediction(value)


@app.template_filter("is_melanoma")
def _is_melanoma_filter(value):
    return is_melanoma(value)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def _analyze_upload():
    if "image" not in request.files:
        flash("Choose an image first.", "danger")
        return redirect(url_for("home"))

    file = request.files["image"]
    if not file or file.filename == "":
        flash("Choose an image first.", "danger")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Use a JPG, PNG, BMP, TIFF, or WebP image.", "danger")
        return redirect(url_for("home"))

    filename = safe_filename(secure_filename(file.filename))
    save_path = config.UPLOAD_DIR / filename
    file.save(str(save_path))

    patient = {
        "patient_id": generate_unique_id("PID"),
        "name": "Anonymous",
        "age": None,
        "sex": "N/A",
    }
    try:
        result = pipeline.run(image_path=str(save_path), patient=patient)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        flash(f"Analysis failed: {exc}", "danger")
        return redirect(url_for("home"))
    return redirect(url_for("prediction", pred_id=result["db_id"]))


@app.route("/", methods=["GET"])
def home():
    user = current_user()
    if user:
        dest = dashboard_endpoint(user)
        if dest:
            return redirect(url_for(dest))
    return render_template("home.html")


@app.route("/about")
def about():
    return redirect(url_for("home", _anchor="about"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return redirect(url_for("home"))
    return _analyze_upload()


@app.route("/prediction/<int:pred_id>")
@login_required
def prediction(pred_id: int):
    record = get_db().get_prediction(pred_id)
    user = current_user()
    if not record or not get_clinic().can_access_case(user, record):
        flash("That result was not found.", "warning")
        dest = dashboard_endpoint(user)
        return redirect(url_for(dest or "home"))
    if user.get("role") == "patient":
        return redirect(url_for("patient.report_detail", case_id=pred_id))
    if user.get("role") == "doctor":
        return redirect(url_for("doctor.case_review", case_id=pred_id))
    images = {
        "original": to_static_url(record.get("image_path")),
        "gradcam": to_static_url(record.get("gradcam_path"))
        if is_melanoma(record.get("prediction"))
        else None,
    }
    return render_template(
        "prediction.html",
        record=record,
        images=images,
        threshold=resolve_decision_threshold(),
        ai_label=display_prediction(record.get("prediction")),
        model_name=display_model_name(record.get("model_used")),
        melanoma=is_melanoma(record.get("prediction")),
    )


@app.route("/prediction/<int:pred_id>/staging", methods=["GET", "POST"])
@role_required("doctor", "admin")
def ajcc_staging(pred_id: int):
    record = get_db().get_prediction(pred_id)
    if not record:
        flash("Prediction not found.", "warning")
        return redirect(url_for("doctor.dashboard"))
    if str(record.get("prediction", "")).lower() != "malignant":
        flash("AJCC staging is only available for malignant predictions.", "warning")
        return redirect(url_for("prediction", pred_id=pred_id))

    if request.method == "GET":
        return render_template(
            "staging.html",
            record=record,
            prefill=None,
            prefill_found=None,
            extract_info=None,
        )

    action = request.form.get("action", "predict")

    if action == "parse_report":
        from staging.clinical_report_parser import (
            ALLOWED_REPORT_EXTENSIONS,
            parse_clinical_report,
        )

        file = request.files.get("clinical_report")
        if not file or not file.filename:
            flash("Please choose a clinical report file.", "danger")
            return redirect(url_for("ajcc_staging", pred_id=pred_id))

        filename = safe_filename(secure_filename(file.filename))
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_REPORT_EXTENSIONS:
            flash("Unsupported report type. Use PDF, TXT, DOCX, or an image.", "danger")
            return redirect(url_for("ajcc_staging", pred_id=pred_id))

        report_dir = config.UPLOAD_DIR / "clinical_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        save_path = report_dir / filename
        file.save(str(save_path))

        try:
            parsed = parse_clinical_report(save_path)
            flash(
                f"Extracted {parsed['fields_found']} clinical field(s). "
                "Review the values below, then predict stage.",
                "success" if parsed["fields_found"] else "warning",
            )
            if parsed["fields_found"] == 0:
                flash(
                    "No AJCC fields were detected. Enter values manually or try a clearer text PDF.",
                    "warning",
                )
            return render_template(
                "staging.html",
                record=record,
                prefill=parsed["values"],
                prefill_found=parsed["found"],
                extract_info=parsed,
            )
        except Exception as exc:
            logger.exception("Clinical report parse failed: %s", exc)
            flash(f"Could not read report: {exc}", "danger")
            return redirect(url_for("ajcc_staging", pred_id=pred_id))

    clinical = {
        "breslow_thickness_mm": request.form.get("breslow_thickness_mm") or 0,
        "ulceration": request.form.get("ulceration", "no"),
        "mitotic_rate": request.form.get("mitotic_rate") or 0,
        "lymph_node_involvement": request.form.get("lymph_node_involvement", "no"),
        "metastasis": request.form.get("metastasis", "no"),
        "num_positive_nodes": request.form.get("num_positive_nodes") or 0,
    }
    doctor_notes = request.form.get("doctor_notes", "")
    try:
        pipeline.apply_ajcc_staging(pred_id, clinical, doctor_notes=doctor_notes)
        flash("AJCC stage predicted. You can print or download the PDF.", "success")
    except Exception as exc:
        logger.exception("AJCC staging failed: %s", exc)
        flash(f"Staging failed: {exc}", "danger")
        return redirect(url_for("ajcc_staging", pred_id=pred_id))

    return redirect(url_for("prediction", pred_id=pred_id))


@app.route("/report/<int:pred_id>")
@login_required
def report(pred_id: int):
    record = get_db().get_prediction(pred_id)
    if not record:
        flash("Report not found.", "warning")
        return redirect(url_for("home"))
    return render_template("report.html", record=record)


@app.route("/report/<int:pred_id>/download")
@login_required
def download_report(pred_id: int):
    record = get_db().get_prediction(pred_id)
    user = current_user()
    if not record or not get_clinic().can_access_case(user, record):
        flash("PDF report unavailable.", "warning")
        dest = dashboard_endpoint(user)
        return redirect(url_for(dest or "home"))
    if user.get("role") == "patient" and record.get("review_status") != "finalized":
        abort(404)
    if not record.get("report_path"):
        flash("PDF file missing on disk.", "danger")
        return redirect(url_for("report", pred_id=pred_id))
    path = Path(record["report_path"])
    if not path.exists():
        flash("PDF file missing on disk.", "danger")
        return redirect(url_for("report", pred_id=pred_id))
    return send_file(str(path), as_attachment=True, download_name=path.name)


@app.route("/history")
@role_required("doctor", "admin")
def history():
    page = max(1, int(request.args.get("page", 1)))
    per_page = 20
    records = get_db().get_history(limit=per_page, offset=(page - 1) * per_page)
    total = get_db().count_predictions()
    return render_template(
        "history.html",
        records=records,
        page=page,
        total=total,
        per_page=per_page,
    )


@app.route("/comparison")
@role_required("doctor", "admin")
def comparison():
    """Model comparison dashboard (bonus)."""
    metrics = {}
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, "r", encoding="utf-8-sig") as fh:
            metrics = json.load(fh)
    return render_template("comparison.html", metrics=metrics)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.route("/api/stats")
@role_required("doctor", "admin")
def api_stats():
    return jsonify(get_db().get_stats())


@app.route("/api/predict", methods=["POST"])
@role_required("doctor", "admin")
def api_predict():
    """JSON/multipart API for programmatic access."""
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file type"}), 400

    filename = safe_filename(secure_filename(file.filename or "upload.jpg"))
    save_path = config.UPLOAD_DIR / filename
    file.save(str(save_path))

    patient = {
        "patient_id": request.form.get("patient_id") or generate_unique_id("PID"),
        "name": request.form.get("patient_name", "API Patient"),
        "age": request.form.get("age"),
        "sex": request.form.get("sex", "N/A"),
    }
    clinical = {
        "breslow_thickness_mm": request.form.get("breslow_thickness_mm", 0),
        "ulceration": request.form.get("ulceration", "no"),
        "mitotic_rate": request.form.get("mitotic_rate", 0),
        "lymph_node_involvement": request.form.get("lymph_node_involvement", "no"),
        "metastasis": request.form.get("metastasis", "no"),
    }
    try:
        result = pipeline.run(str(save_path), patient, clinical)
        # Make JSON-serializable
        result.pop("gradcam", None)
        return jsonify(result)
    except Exception as exc:
        logger.exception("API predict failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/reports/generated/<path:filename>")
def serve_report_file(filename):
    return send_from_directory(str(config.REPORTS_DIR), filename)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum size is 16 MB.", "danger")
    return redirect(url_for("doctor.analyze")), 413


@app.errorhandler(403)
def forbidden(e):
    flash("You are not allowed to open that page.", "danger")
    user = current_user()
    dest = dashboard_endpoint(user) if user else None
    return redirect(url_for(dest or "auth.login"))


@app.errorhandler(404)
def not_found(e):
    user = current_user()
    dest = dashboard_endpoint(user) if user else None
    if dest:
        return redirect(url_for(dest))
    return render_template("home.html"), 404


def create_app() -> Flask:
    get_clinic()
    get_db()
    return app


if __name__ == "__main__":
    get_clinic()
    get_db()
    logger.info(
        "Starting Melanoma CDSS on http://%s:%s",
        config.FLASK_HOST,
        config.FLASK_PORT,
    )
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
