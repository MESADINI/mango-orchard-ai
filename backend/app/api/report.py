"""
AI Mango Orchard Health & Pest Intelligence — Scan Report Endpoint
Renders a stored scan as HTML and converts it to PDF with WeasyPrint. When WeasyPrint's
native libraries (Pango/Cairo) are unavailable, the same HTML is returned instead.
"""

import logging
from html import escape
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Response

from app.core.deps import get_current_user, get_store
from app.core.store import Store

logger = logging.getLogger(__name__)
router = APIRouter()

REPORT_CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1e293b; line-height: 1.45; font-size: 12px; }
.header { border-bottom: 3px solid #059669; padding-bottom: 10px; margin-bottom: 18px; }
.logo { font-size: 20px; font-weight: bold; color: #059669; }
.muted { color: #64748b; }
.card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 16px; }
.score { font-size: 34px; font-weight: 800; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 9999px; background: #f1f5f9; font-weight: bold; font-size: 11px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 16px 0; }
th, td { border: 1px solid #cbd5e1; padding: 6px 9px; text-align: left; font-size: 11px; vertical-align: top; }
th { background: #f1f5f9; }
h3 { font-size: 13px; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; margin-top: 18px; }
.caveat { color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 6px 9px; margin: 4px 0; font-size: 11px; }
.citation { font-size: 10px; color: #64748b; font-style: italic; }
.footer { margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 8px; font-size: 10px; color: #94a3b8; text-align: center; }
"""


def _rows(rows: List[List[Any]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)


def _num(value) -> str:
    """Formats a penalty value; None means the component was not assessed."""
    return "not assessed" if value is None else f"{value:.2f}"

def _empty_row(message: str, colspan: int) -> str:
    return f'<tr><td colspan="{colspan}" class="muted">{escape(message)}</td></tr>'


def render_report_html(scan: Dict[str, Any]) -> str:
    e = lambda v: escape(str(v))  # noqa: E731

    model_status = scan.get("model_status", {})
    detection_rows = [
        [
            "Foliar Disease",
            e(d["disease_name"]),
            f"{d['confidence'] * 100:.1f}%",
            f"{d['severity_percentage']:.1f}% of leaf surface",
            "EfficientNet-B3",
        ]
        for d in scan.get("disease_results", [])
    ]
    detection_rows += [
        [
            "Canopy Pest",
            e(p["pest_name"]),
            f"{p['confidence'] * 100:.1f}%",
            "bbox " + e(", ".join(f"{v:.0f}" for v in p["bbox"])),
            "UNVERIFIED (Rule 3)" if p.get("is_unreliable") else "YOLOv8",
        ]
        for p in scan.get("pest_results", [])
    ]
    detections_html = _rows(detection_rows) or _empty_row("No disease or pest detections in this scan.", 5)

    rec_html = _rows([
        [
            e(r.get("condition") or r.get("title")),
            e(r["treatment_type"]),
            e(r["product_name"]),
            f"{e(r['dosage_rate'])}. {e(r['application_guidance'])} "
            + (f"(Last spray at least {int(r['waiting_period_days'])} days before harvest)"
             if r.get("waiting_period_days") is not None else "(Check the product label for the pre-harvest interval)"),
            f'<span class="citation">{e(r["citation_source"])}</span>',
        ]
        for r in scan.get("recommendations", [])
    ]) or _empty_row("No treatment protocol required for the detected conditions.", 5)

    penalties = scan.get("penalty_breakdown", {})
    weights = scan.get("weights_used", {})
    penalty_html = _rows([
        ["Disease severity (S<sub>d</sub>)", _num(penalties.get("disease_severity_penalty")), f"{weights.get('disease', 0):.2f}"],
        ["Pest pressure (P<sub>p</sub>)", _num(penalties.get("pest_pressure_penalty")), f"{weights.get('pest', 0):.2f}"],
        ["Weather vulnerability (W<sub>v</sub>)", _num(penalties.get("weather_vulnerability_index")), f"{weights.get('weather', 0):.2f}"],
        ["Nutrient deficiency (N<sub>d</sub>)", _num(penalties.get("nutrient_penalty")), f"{weights.get('nutrient', 0):.2f}"],
        ["<strong>Weighted total penalty</strong>", f"<strong>{_num(penalties.get('weighted_total_penalty'))}</strong>", ""],
    ])

    weather = scan.get("weather_vulnerability", {})
    if weather.get("live_data"):
        weather_line = (
            f"{weather['current_temperature_c']:.1f}°C, {weather['current_relative_humidity_pct']:.0f}% RH, "
            f"{weather['current_precipitation_mm']:.1f} mm rain (live Open-Meteo data)"
        )
    else:
        weather_line = "Live weather data unavailable; the weather component was not included."
    score = scan.get("ohi_score")
    score_text = "—" if score is None else f"{score:.1f} / 100"

    caveats_html = "".join(f'<div class="caveat">{e(c)}</div>' for c in scan.get("caveats", []))
    color = e(scan.get("badge_color", "#1e293b"))
    partial_note = (
        f'<div class="muted">Partial index: {e(", ".join(scan.get("ohi_missing_components", [])))} component(s) unavailable.</div>'
        if scan.get("ohi_is_partial") else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Orchard Health Report - {e(scan['scan_id'])}</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="header">
  <div class="logo">AI Mango Orchard Health &amp; Pest Intelligence</div>
  <div class="muted">Scan report &middot; ID {e(scan['scan_id'])} &middot; {e(scan.get('created_at', ''))}</div>
</div>

<div class="card">
  <div class="muted" style="font-weight:600;">ORCHARD HEALTH INDEX (OHI)</div>
  <div class="score" style="color:{color};">{score_text}</div>
  <span class="badge" style="color:{color};">{e(scan['health_status'])}</span>
  {partial_note}
  <div class="muted" style="margin-top:6px;">Location: {scan.get('latitude', 0):.4f}, {scan.get('longitude', 0):.4f}
   &middot; Scan type: {e(scan.get('scan_type', ''))} &middot; Weather: {e(weather_line)}</div>
  <div class="muted">Models: disease {'trained' if model_status.get('disease_model_trained') else 'NOT INSTALLED'},
   pest {'trained' if model_status.get('pest_model_trained') else 'NOT INSTALLED'}</div>
</div>

{('<h3>Caveats &amp; Disclosures</h3>' + caveats_html) if caveats_html else ''}

<h3>Diagnosis Breakdown</h3>
<table>
  <thead><tr><th>Domain</th><th>Diagnosis / Target</th><th>Confidence</th><th>Severity / Location</th><th>Source</th></tr></thead>
  <tbody>{detections_html}</tbody>
</table>

<h3>OHI Penalty Breakdown</h3>
<table>
  <thead><tr><th>Component</th><th>Penalty (0-100)</th><th>Weight</th></tr></thead>
  <tbody>{penalty_html}</tbody>
</table>

<h3>Verified Treatment Recommendations</h3>
<table>
  <thead><tr><th>Target</th><th>Type</th><th>Product</th><th>Dosage &amp; Guidance</th><th>Source Citation</th></tr></thead>
  <tbody>{rec_html}</tbody>
</table>

<div class="footer">Generated by AI Mango Orchard Health &amp; Pest Intelligence. Decision support only;
confirm diagnoses in the field and follow product labels and local extension advice.</div>
</body>
</html>"""


def html_to_pdf(html_content: str):
    """Returns PDF bytes, or None when WeasyPrint or its native libraries are unavailable."""
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        logger.warning("WeasyPrint unavailable (%s); serving HTML report instead.", e)
        return None
    return HTML(string=html_content).write_pdf()


@router.get("/generate/{scan_id}")
def generate_pdf_report(scan_id: str, user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Generates a PDF report for a stored scan (HTML if WeasyPrint's native libraries are missing)."""
    scan = store.get_scan(user["id"], scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan ID {scan_id} not found.")

    html_content = render_report_html(scan)
    pdf = html_to_pdf(html_content)
    if pdf is not None:
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="mango_orchard_report_{scan_id}.pdf"'},
        )
    return Response(
        content=html_content,
        media_type="text/html; charset=utf-8",
        headers={"X-Report-Fallback": "WeasyPrint native libraries not found; rendered HTML"},
    )
