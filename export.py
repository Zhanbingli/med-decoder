#!/usr/bin/env python3
"""
Note export — text / Markdown / PDF
===================================

Get the finished note OUT of the app and into the chart. Plain text and
Markdown are dependency-free; PDF uses reportlab with a built-in CJK font
(no font files to bundle), so bilingual labels render correctly.
"""

import io
import time
from typing import Dict, List, Optional, Tuple

# (key, English / 中文 label) in display order
NOTE_SECTIONS: List[Tuple[str, str]] = [
    ("chief_complaint", "Chief Complaint / 主诉"),
    ("present_history", "History of Present Illness / 现病史"),
    ("past_history", "Past Medical History / 既往史"),
    ("cardiovascular_exam", "Examination / 查体"),
    ("ecg_findings", "ECG / 心电图"),
    ("assessment", "Assessment / 评估"),
    ("plan", "Plan / 计划"),
]

_DISCLAIMER = (
    "AI-assisted draft generated from the visit transcript. "
    "Empty sections were not documented in the conversation. "
    "Reviewed and verified by a clinician before clinical use."
)


def _header_lines(patient: Optional[Dict], meta: Optional[Dict]) -> List[str]:
    patient = patient or {}
    meta = meta or {}
    out = []
    name = patient.get("name") or "—"
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    out.append(f"Patient / 患者: {name}    {age} {gender}".rstrip())
    out.append(f"Date / 日期: {meta.get('date', time.strftime('%Y-%m-%d %H:%M'))}")
    if meta.get("template"):
        out.append(f"Template / 模板: {meta['template']}")
    if meta.get("status"):
        out.append(f"Status / 状态: {meta['status']}")
    if meta.get("verified_by"):
        out.append(f"Verified by / 核验: {meta['verified_by']}")
    return out


def format_text(fields: Dict[str, str], patient=None, meta=None) -> str:
    """Plain-text note suitable for pasting into an EHR."""
    lines = ["OUTPATIENT NOTE / 门诊病历", "=" * 40]
    lines += _header_lines(patient, meta)
    lines.append("-" * 40)
    for key, label in NOTE_SECTIONS:
        val = (fields.get(key) or "").strip()
        lines.append(f"\n{label}:")
        lines.append(f"  {val if val else '(not documented / 未提供)'}")
    lines.append("\n" + "-" * 40)
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def format_markdown(fields: Dict[str, str], patient=None, meta=None) -> str:
    lines = ["# Outpatient Note / 门诊病历", ""]
    for h in _header_lines(patient, meta):
        lines.append(f"- {h}")
    lines.append("")
    for key, label in NOTE_SECTIONS:
        val = (fields.get(key) or "").strip()
        lines.append(f"## {label}")
        lines.append(val if val else "_(not documented / 未提供)_")
        lines.append("")
    lines.append("---")
    lines.append(f"_{_DISCLAIMER}_")
    return "\n".join(lines)


def format_pdf(fields: Dict[str, str], patient=None, meta=None) -> bytes:
    """Render the note to PDF bytes (reportlab, CJK-capable)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    font = "STSong-Light"  # built-in CJK font
    pdfmetrics.registerFont(UnicodeCIDFont(font))

    title = ParagraphStyle("title", fontName=font, fontSize=15, spaceAfter=8, leading=19)
    meta_s = ParagraphStyle("meta", fontName=font, fontSize=9, textColor="#555555", leading=13)
    label_s = ParagraphStyle("label", fontName=font, fontSize=10.5, spaceBefore=8,
                             spaceAfter=2, textColor="#0E7C7B", leading=14)
    body_s = ParagraphStyle("body", fontName=font, fontSize=10.5, leading=15)
    foot_s = ParagraphStyle("foot", fontName=font, fontSize=8, textColor="#888888",
                            leading=11, spaceBefore=10)

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm, title="Outpatient Note",
    )
    story = [Paragraph("Outpatient Note / 门诊病历", title)]
    for h in _header_lines(patient, meta):
        story.append(Paragraph(esc(h), meta_s))
    story.append(Spacer(1, 6))
    for key, label in NOTE_SECTIONS:
        val = (fields.get(key) or "").strip()
        story.append(Paragraph(esc(label), label_s))
        story.append(Paragraph(esc(val) if val else
                               "<i>(not documented / 未提供)</i>", body_s))
    story.append(Paragraph(_DISCLAIMER, foot_s))
    doc.build(story)
    return buf.getvalue()
