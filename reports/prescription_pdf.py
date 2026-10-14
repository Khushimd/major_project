"""Prescription PDF — issued only by authenticated doctors."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, HRFlowable

import config


def generate_prescription_pdf(
    prescription: Dict[str, Any],
    items: List[Dict[str, Any]],
    output_dir: Path = config.REPORTS_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code = prescription.get("code") or f"RX-{prescription.get('id')}"
    path = output_dir / f"{code}.pdf"

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "RxTitle", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#0B3D4F"), spaceAfter=8
    )
    body = ParagraphStyle("RxBody", parent=styles["Normal"], fontSize=10, leading=14)

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm)
    story = [
        Paragraph("Smart Skin Health — Prescription", title),
        Paragraph("Issued by an authenticated clinician. AI does not generate prescriptions.", body),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B3D4F")),
        Spacer(1, 10),
        Paragraph(f"<b>Prescription ID:</b> {code}", body),
        Paragraph(f"<b>Date:</b> {prescription.get('created_at') or datetime.now().strftime('%Y-%m-%d %H:%M')}", body),
        Paragraph(f"<b>Patient:</b> {prescription.get('patient_name', '')}", body),
        Paragraph(f"<b>Doctor:</b> {prescription.get('doctor_name', '')}", body),
        Paragraph(f"<b>Diagnosis:</b> {prescription.get('diagnosis') or '—'}", body),
        Spacer(1, 12),
        Paragraph("<b>Medicines</b>", body),
        Spacer(1, 6),
    ]
    data = [["Medicine", "Dosage", "Frequency", "Duration", "Instructions"]]
    for item in items:
        data.append(
            [
                item.get("medicine") or "",
                item.get("dosage") or "",
                item.get("frequency") or "",
                item.get("duration") or "",
                item.get("instructions") or "",
            ]
        )
    table = Table(data, colWidths=[1.6 * inch, 1.1 * inch, 1.2 * inch, 1.1 * inch, 1.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F1F5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#0B3D4F")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    if prescription.get("notes"):
        story.extend(
            [Spacer(1, 12), Paragraph(f"<b>Notes:</b> {prescription.get('notes')}", body)]
        )
    story.extend(
        [
            Spacer(1, 20),
            Paragraph(config.MEDICAL_DISCLAIMER, body),
        ]
    )
    doc.build(story)
    return path
