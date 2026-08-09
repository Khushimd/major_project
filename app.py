"""
Flask web application — Melanoma Clinical Decision Support System.

Pages: Home | About | Upload | Prediction | Report | History | Comparison
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from database.db import get_db
from utils.helpers import allowed_file, generate_unique_id, safe_filename
from utils.logger import setup_logger
from utils.pipeline import pipeline

logger = setup_logger("app")

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)
app.secret_key = config.FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(config.UPLOAD_DIR)


@app.context_processor
def inject_globals():
    return {
        "disclaimer": config.MEDICAL_DISCLAIMER,
        "app_title": "Melanoma CDSS",
        "full_title": (
            "AI-Powered Clinical Decision Support System "
            "for Melanoma Detection & AJCC Stage Prediction"
        ),
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    stats = get_db().get_stats()
    metrics = {}
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, "r", encoding="utf-8-sig") as fh:
            metrics = json.load(fh)
    return render_template("home.html", stats=stats, metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    if "image" not in request.files:
        flash("No image file provided.", "danger")
        return redirect(url_for("upload"))

    file = request.files["image"]
    if not file or file.filename == "":
        flash("Please select an image to upload.", "danger")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff, webp.", "danger")
        return redirect(url_for("upload"))

    filename = safe_filename(secure_filename(file.filename))
    save_path = config.UPLOAD_DIR / filename
    file.save(str(save_path))

    patient = {
        "patient_id": request.form.get("patient_id") or generate_unique_id("PID"),
        "name": request.form.get("patient_name", "").strip() or "Anonymous",
        "age": request.form.get("age") or None,
        "sex": request.form.get("sex", "N/A"),
    }
    try:
        if patient["age"] is not None and patient["age"] != "":
            patient["age"] = int(patient["age"])
    except ValueError:
        patient["age"] = None

    clinical = None  # AJCC clinical values are collected after a malignant result
    doctor_notes = ""

    try:
        result = pipeline.run(
            image_path=str(save_path),
            patient=patient,
            clinical=clinical,
            doctor_notes=doctor_notes,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        flash(f"Analysis failed: {exc}", "danger")
        return redirect(url_for("upload"))

    return redirect(url_for("prediction", pred_id=result["db_id"]))


@app.route("/prediction/<int:pred_id>")
def prediction(pred_id: int):
    record = get_db().get_prediction(pred_id)
    if not record:
        flash("Prediction not found.", "warning")
        return redirect(url_for("history"))

    # Build relative static URLs for images
    def to_static_url(path):
        if not path:
            return None
        p = Path(path)
        try:
            rel = p.relative_to(ROOT / "static")
            return url_for("static", filename=str(rel).replace("\\", "/"))
        except ValueError:
            return None

    images = {
        "original": to_static_url(record.get("image_path")),
        "gradcam": to_static_url(record.get("gradcam_path")),
    }
    return render_template("prediction.html", record=record, images=images)


@app.route("/prediction/<int:pred_id>/staging", methods=["GET", "POST"])
def ajcc_staging(pred_id: int):
    record = get_db().get_prediction(pred_id)
    if not record:
        flash("Prediction not found.", "warning")
        return redirect(url_for("history"))
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
def report(pred_id: int):
    record = get_db().get_prediction(pred_id)
    if not record:
        flash("Report not found.", "warning")
        return redirect(url_for("history"))
    return render_template("report.html", record=record)


@app.route("/report/<int:pred_id>/download")
def download_report(pred_id: int):
    record = get_db().get_prediction(pred_id)
    if not record or not record.get("report_path"):
        flash("PDF report unavailable.", "warning")
        return redirect(url_for("history"))
    path = Path(record["report_path"])
    if not path.exists():
        flash("PDF file missing on disk.", "danger")
        return redirect(url_for("report", pred_id=pred_id))
    return send_file(str(path), as_attachment=True, download_name=path.name)


@app.route("/history")
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
def api_stats():
    return jsonify(get_db().get_stats())


@app.route("/api/predict", methods=["POST"])
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
    return redirect(url_for("upload")), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("home.html", stats=get_db().get_stats(), metrics={}), 404


def create_app() -> Flask:
    get_db()  # ensure DB exists
    return app


if __name__ == "__main__":
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
