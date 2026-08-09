"""
PDF diagnostic report generator for Melanoma CDSS.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

import config
from utils.logger import setup_logger

logger = setup_logger("reports")


class DiagnosticReportGenerator:
    """Generate a professional multi-section PDF diagnostic report."""

    def __init__(self, output_dir: Path = config.REPORTS_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = self._build_styles()

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        styles = {
            "title": ParagraphStyle(
                "ReportTitle",
                parent=base["Heading1"],
                fontSize=18,
                textColor=colors.HexColor("#0B3D4F"),
                alignment=TA_CENTER,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "subtitle": ParagraphStyle(
                "Subtitle",
                parent=base["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#4A6670"),
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "heading": ParagraphStyle(
                "SectionHead",
                parent=base["Heading2"],
                fontSize=12,
                textColor=colors.HexColor("#0B3D4F"),
                spaceBefore=14,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "Body",
                parent=base["Normal"],
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
            ),
            "disclaimer": ParagraphStyle(
                "Disclaimer",
                parent=base["Normal"],
                fontSize=8,
                textColor=colors.HexColor("#8B1E1E"),
                alignment=TA_CENTER,
                leading=11,
            ),
            "notes": ParagraphStyle(
                "Notes",
                parent=base["Normal"],
                fontSize=10,
                leading=16,
                textColor=colors.HexColor("#333333"),
            ),
            "label": ParagraphStyle(
                "Label",
                parent=base["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#555555"),
            ),
        }
        return styles

    def generate(
        self,
        patient: Dict[str, Any],
        prediction: Dict[str, Any],
        staging: Optional[Dict[str, Any]] = None,
        gradcam_path: Optional[str] = None,
        lesion_image_path: Optional[str] = None,
        doctor_notes: str = "",
        report_id: Optional[str] = None,
    ) -> Path:
        """
        Build and write the PDF report.

        Returns:
            Path to the generated PDF file.
        """
        report_id = report_id or f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        out_path = self.output_dir / f"{report_id}.pdf"

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title=f"Melanoma CDSS Report — {report_id}",
        )

        story = []
        story.append(Paragraph(
            "AI-Powered Clinical Decision Support System",
            self.styles["title"],
        ))
        story.append(Paragraph(
            "Melanoma Detection &amp; AJCC Stage Prediction — Diagnostic Report",
            self.styles["subtitle"],
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0B3D4F")))
        story.append(Spacer(1, 8))

        # Disclaimer
        story.append(Paragraph(config.MEDICAL_DISCLAIMER, self.styles["disclaimer"]))
        story.append(Spacer(1, 10))

        # Patient details
        story.append(Paragraph("1. Patient Details", self.styles["heading"]))
        patient_rows = [
            ["Report ID", report_id],
            ["Patient ID", str(patient.get("patient_id", "N/A"))],
            ["Patient Name", str(patient.get("name", "N/A"))],
            ["Age", str(patient.get("age", "N/A"))],
            ["Sex", str(patient.get("sex", "N/A"))],
            ["Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ]
        story.append(self._kv_table(patient_rows))

        # Prediction
        story.append(Paragraph("2. Deep Learning Prediction", self.styles["heading"]))
        pred_label = prediction.get("prediction", "N/A")
        conf_pct = round(float(prediction.get("confidence", 0)) * 100, 2)
        pred_rows = [
            ["Classification", pred_label],
            ["Confidence Score", f"{conf_pct}%"],
            ["Malignant Probability", f"{float(prediction.get('malignant_prob', 0))*100:.2f}%"],
            ["Benign Probability", f"{float(prediction.get('benign_prob', 0))*100:.2f}%"],
            ["Model Used", str(prediction.get("model_used", "Ensemble"))],
        ]
        if prediction.get("individual"):
            for k, v in prediction["individual"].items():
                pred_rows.append([f"{k.title()} Prob.", f"{float(v)*100:.2f}%"])
        story.append(self._kv_table(pred_rows))

        # Images
        story.append(Paragraph("3. Imaging", self.styles["heading"]))
        if lesion_image_path and Path(lesion_image_path).exists():
            story.extend(self._safe_image(lesion_image_path, "Lesion Image"))
        if gradcam_path and Path(gradcam_path).exists():
            story.extend(self._safe_image(gradcam_path, "Grad-CAM Overlay"))
        elif pred_label.lower() == "benign":
            story.append(Paragraph(
                "<b>Grad-CAM:</b> No malignant regions detected.",
                self.styles["body"],
            ))

        # AJCC Staging
        story.append(Paragraph("4. AJCC Stage Prediction", self.styles["heading"]))
        if staging and pred_label.lower() == "malignant":
            stage_rows = [
                ["AJCC Stage", staging.get("stage", "N/A")],
                ["TNM", f"{staging.get('t_category','')} {staging.get('n_category','')} {staging.get('m_category','')}"],
                ["Risk Level", staging.get("risk_level", "N/A")],
                ["Details", staging.get("details", "")],
            ]
            story.append(self._kv_table(stage_rows))
            story.append(Spacer(1, 6))
            story.append(Paragraph("5. Clinical Recommendation", self.styles["heading"]))
            story.append(Paragraph(staging.get("recommendation", ""), self.styles["body"]))
            story.append(Paragraph("6. Suggested Follow-up", self.styles["heading"]))
            story.append(Paragraph(staging.get("follow_up", ""), self.styles["body"]))
        elif pred_label.lower() == "malignant":
            story.append(Paragraph(
                "Malignant classification confirmed. AJCC staging pending clinical parameters "
                "(Breslow thickness, ulceration, mitotic rate, lymph nodes, metastasis).",
                self.styles["body"],
            ))
            story.append(Paragraph("5. Clinical Recommendation", self.styles["heading"]))
            story.append(Paragraph(
                "Complete AJCC clinical form in the CDSS to generate stage-specific guidance.",
                self.styles["body"],
            ))
            story.append(Paragraph("6. Suggested Follow-up", self.styles["heading"]))
            story.append(Paragraph(
                "Urgent dermatology / oncology review pending formal staging.",
                self.styles["body"],
            ))
        else:
            story.append(Paragraph(
                "AJCC staging not applicable (lesion classified as Benign).",
                self.styles["body"],
            ))
            story.append(Paragraph("5. Clinical Recommendation", self.styles["heading"]))
            story.append(Paragraph(
                "Routine dermatologic monitoring. Reassurance with sun-safety education. "
                "Re-evaluate if lesion changes in size, color, or border irregularity.",
                self.styles["body"],
            ))
            story.append(Paragraph("6. Suggested Follow-up", self.styles["heading"]))
            story.append(Paragraph(
                "Annual skin examination or sooner if clinical concern arises.",
                self.styles["body"],
            ))

        # Doctor notes
        story.append(Paragraph("7. Doctor Notes", self.styles["heading"]))
        notes = doctor_notes.strip() if doctor_notes else "_" * 80
        story.append(Paragraph(notes.replace("\n", "<br/>"), self.styles["notes"]))
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "Physician Signature: ________________________ &nbsp;&nbsp;&nbsp; Date: ______________",
            self.styles["body"],
        ))
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Paragraph(config.MEDICAL_DISCLAIMER, self.styles["disclaimer"]))

        doc.build(story)
        logger.info("PDF report written: %s", out_path)
        return out_path

    def _kv_table(self, rows: list) -> Table:
        data = [[Paragraph(f"<b>{k}</b>", self.styles["label"]),
                 Paragraph(str(v), self.styles["body"])] for k, v in rows]
        table = Table(data, colWidths=[2.2 * inch, 4.3 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F1F5")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#0B3D4F")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0C4CE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def _safe_image(self, path: str, caption: str) -> list:
        elements = [Paragraph(f"<b>{caption}</b>", self.styles["label"]), Spacer(1, 4)]
        try:
            img = Image(path, width=3.2 * inch, height=3.2 * inch, kind="proportional")
            elements.append(img)
            elements.append(Spacer(1, 8))
        except Exception as exc:
            logger.warning("Could not embed image %s: %s", path, exc)
            elements.append(Paragraph(f"[Image unavailable: {path}]", self.styles["body"]))
        return elements
