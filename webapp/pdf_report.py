"""Build a user-facing PDF from an analysis job's plain-text report.

The analysis scripts write a technical report (frame counts, raw violation
counters, file paths). This module keeps only what matters to the person who
did the test: their result, how good their form was, what to work on, and how
to record a better video next time.
"""
import os
import re
import time
from collections import OrderedDict

from fpdf import FPDF

FONT_DIR = "/System/Library/Fonts/Supplemental"
INK = (29, 33, 31)
MUTED = (107, 111, 106)
LINE = (226, 221, 211)
ACCENT = (47, 93, 70)
TINT = (233, 239, 233)
PAPER = (247, 245, 240)
WARM = (163, 64, 47)

KV = re.compile(r"^\s*[-•*]?\s*([A-Za-z][A-Za-z0-9 _\-/()%.]{1,48}):\s*(.+?)\s*$")
NUM = re.compile(r"-?\d+(?:\.\d+)?")

# Per-test wording for the main result.
RESULT = {
    "pushup": ("Valid push-ups", "{:.0f}", ""),
    "plank": ("Plank held", "{:.1f}", " s"),
    "verticaljump": ("Highest jump", "{:.0f}", " cm"),   # report gives metres; converted below
    "shuttlerun": ("Average speed", "{:.2f}", " m/s"),
}

FILMING = {
    "pushup": "Film from the side at floor level, with your whole body from head to heels in frame.",
    "squats": "Film from the side with your whole body in frame, including your feet.",
    "curlups": "Film from the side. Keep your knees bent and feet flat on the floor.",
    "plank": "Film from the side at floor level so your shoulders, hips and ankles are all visible.",
    "jumpingjack": "Film from the front. Leave room above your hands and below your feet.",
    "verticaljump": "Film from the side with a steady camera. Keep your feet in frame for the whole jump.",
    "shuttlerun": "Film from the side, far enough back to see both turning markers.",
    "sitreach": "Film from the side. Keep your legs straight and reach forward slowly.",
    "backstretch": "Film from the side and bend forward slowly from the hips.",
}

# Push-up form issues: report label -> (plain description, tip)
PUSHUP_ISSUES = OrderedDict([
    ("arm asymmetry issues", ("Arms moving unevenly", "Lower and push with both arms at the same speed. Place your hands the same distance from your body.")),
    ("body alignment issues", ("Hips sagging or piking", "Squeeze your glutes and brace your core so your body stays in one straight line.")),
    ("leg position issues", ("Knees bending", "Keep your legs straight and your weight on your toes.")),
    ("head position issues", ("Head lifting or dropping", "Look at the floor slightly ahead of your hands to keep your neck neutral.")),
])

# Plank / jump flagged events -> plain wording
EVENT_TEXT = {
    "sagging": ("Hips sagging", "Tighten your core and glutes to lift your hips back in line."),
    "piked": ("Hips too high", "Lower your hips until shoulders, hips and ankles line up."),
    "bent knees": ("Knees bending", "Keep your legs straight and push back through your heels."),
    "arched neck": ("Neck arched", "Keep your gaze on the floor just in front of your hands."),
    "lateral shift": ("Body shifting sideways", "Spread your weight evenly across both forearms."),
    "knee not flexed enough": ("Not enough knee bend before jumping", "Dip into a quarter squat before you jump to use your legs fully."),
    "improper landing": ("Unsteady landing", "Land softly on both feet with knees slightly bent."),
    "camera movement detected": ("Camera moved", "Place the phone on a stable surface or tripod."),
}


def _clean(text):
    return re.sub(r"[\U00010000-\U0010FFFF\uFE0F\u200D]", "", text or "").strip()


def _num(value):
    m = NUM.search(value or "")
    return float(m.group()) if m else None


def _parse(text):
    """Return (fields, sections). fields: lower-case key -> value. sections: heading -> list of lines."""
    fields, sections, heading = {}, OrderedDict(), ""
    lines = [_clean(l) for l in text.splitlines()]
    for i, line in enumerate(lines):
        if not line or set(line) <= set("=-_"):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (nxt and set(nxt) <= set("=")) or (line.endswith(":") and line[:-1].isupper()):
            heading = line.rstrip(":").strip().lower()
            sections.setdefault(heading, [])
            continue
        sections.setdefault(heading, []).append(line)
        m = KV.match(line)
        if m:
            fields.setdefault(m.group(1).replace("_", " ").strip().lower(), m.group(2))
    return fields, sections


def _rating(score):
    if score is None:
        return None
    if score >= 3.5:
        return "Excellent"
    if score >= 2.5:
        return "Good"
    if score >= 1.5:
        return "Fair"
    return "Needs work"


def _summarise(job):
    """Pull the user-relevant content out of the raw report."""
    key = job.get("test_key") or ""
    fields, sections = _parse(job.get("report") or "")
    out = {"main": None, "highlights": [], "issues": [], "tips": [], "duration": None, "form": None}

    out["duration"] = _num(fields.get("session duration"))
    form = _num(fields.get("average form score") or fields.get("form score"))
    out["form"] = form

    # Main result
    value = _num(fields.get("value"))
    if key in RESULT and value is not None:
        label, fmt, unit = RESULT[key]
        if key == "verticaljump":
            value *= 100
        out["main"] = (label, fmt.format(value) + unit)

    # Highlights: a few plain-language figures per test
    if key == "pushup":
        attempts = _num(fields.get("total pushup attempts"))
        rate = fields.get("success rate")
        consistency = _num(fields.get("form consistency"))
        if attempts is not None:
            out["highlights"].append(("Push-ups attempted", f"{attempts:.0f}"))
        if rate:
            out["highlights"].append(("Counted as valid", rate))
        if consistency is not None:
            out["highlights"].append(("Time in good form", f"{consistency:.0f}%"))
        if out["duration"] and value:
            out["highlights"].append(("Pace", f"{value / out['duration'] * 60:.0f} per minute"))
    elif key == "plank":
        if out["duration"]:
            out["highlights"].append(("Video length", f"{out['duration']:.0f} s"))
    elif key == "verticaljump":
        for k, label in (("total jumps", "Jumps recorded"), ("landing stability", "Landing")):
            if fields.get(k):
                out["highlights"].append((label, fields[k]))
    elif key == "shuttlerun":
        for k, label in (("total runs", "Runs recorded"), ("speed trend", "Speed across runs")):
            if fields.get(k):
                out["highlights"].append((label, fields[k]))

    # What to work on
    if key == "pushup":
        total = _num(fields.get("total frames analyzed")) or 0
        for raw, (desc, tip) in PUSHUP_ISSUES.items():
            n = _num(fields.get(raw)) or 0
            share = n / total * 100 if total else 0
            if share >= 5:
                out["issues"].append((desc, f"{share:.0f}% of the session"))
                out["tips"].append(tip)
    for heading in ("cheats detected",):
        counts = OrderedDict()
        for line in sections.get(heading, []):
            m = re.match(r"^-?\s*([\d.]+)s:\s*(.+)$", line)
            kind = (m.group(2) if m else line.lstrip("- ")).strip().lower()
            first = float(m.group(1)) if m else None
            if kind not in counts:
                counts[kind] = [0, first]
            counts[kind][0] += 1
        for kind, (n, first) in counts.items():
            desc, tip = EVENT_TEXT.get(kind, (kind.capitalize(), None))
            when = f"first at {first:.0f} s" if first is not None else ""
            out["issues"].append((desc, f"{n} time{'s' if n != 1 else ''}" + (f", {when}" if when else "")))
            if tip:
                out["tips"].append(tip)

    # Keep the script's own recommendations, minus duplicates
    for line in sections.get("recommendations", []):
        if line.startswith(("- ", "• ", "* ")):
            tip = line[2:].strip()
            if tip and tip not in out["tips"]:
                out["tips"].append(tip)
    return out


class Report(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Sans", size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "FitCheck. Results are estimates from video and are not medical advice.", align="L")
        self.cell(0, 6, f"{self.page_no()} / {{nb}}", align="R")


def _section(pdf, title):
    if pdf.get_y() > pdf.h - 55:
        pdf.add_page()
    pdf.ln(4)
    pdf.set_font("Sans", "B", size=9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, title.upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(2.5)


def _row(pdf, label, value, value_color=INK):
    w = pdf.epw
    pdf.set_font("Sans", size=10.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(w * 0.6, 7.5, label)
    pdf.set_text_color(*value_color)
    pdf.cell(w * 0.4, 7.5, value, align="R", new_x="LMARGIN", new_y="NEXT")


def build_pdf(job, job_dir, out_path):
    pdf = Report(format="A4", unit="mm")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(True, margin=22)
    pdf.add_font("Sans", "", os.path.join(FONT_DIR, "Arial Unicode.ttf"))
    pdf.add_font("Sans", "B", os.path.join(FONT_DIR, "Arial Bold.ttf"))
    pdf.add_font("Serif", "", os.path.join(FONT_DIR, "Georgia.ttf"))
    pdf.alias_nb_pages()
    pdf.add_page()
    w = pdf.epw
    s = _summarise(job)

    # Title
    created = time.strftime("%d %B %Y, %H:%M", time.localtime(job.get("created") or time.time()))
    pdf.set_font("Serif", size=11)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 6, "FitCheck", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Serif", size=26)
    pdf.set_text_color(*INK)
    pdf.cell(0, 13, job.get("test", "Fitness test"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Sans", size=10)
    pdf.set_text_color(*MUTED)
    meta = created + (f"  ·  {s['duration']:.0f} second recording" if s["duration"] else "")
    pdf.cell(0, 6, meta, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Result cards: main result + form rating
    cards = []
    if s["main"]:
        cards.append((s["main"][1], s["main"][0], True))
    rating = _rating(s["form"])
    if rating:
        cards.append((rating, f"Form  ·  {s['form']:.1f} out of 4", False))
    if cards:
        y0, cw, gap = pdf.get_y(), (w - 4 * (len(cards) - 1)) / len(cards), 4
        for i, (big, small, primary) in enumerate(cards):
            x = pdf.l_margin + i * (cw + gap)
            pdf.set_fill_color(*(TINT if primary else PAPER))
            pdf.rect(x, y0, cw, 28, style="F")
            pdf.set_xy(x + 6, y0 + 5)
            pdf.set_font("Serif", size=22)
            pdf.set_text_color(*(ACCENT if primary else INK))
            pdf.cell(cw - 12, 11, big)
            pdf.set_xy(x + 6, y0 + 17)
            pdf.set_font("Sans", size=9.5)
            pdf.set_text_color(*MUTED)
            pdf.cell(cw - 12, 6, small)
        pdf.set_y(y0 + 34)
    elif not job.get("report"):
        pdf.set_font("Sans", size=10.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(w, 6.5, "This test does not produce a scored report yet. "
                               "Watch the analysed video on FitCheck to see your movement with the pose overlay.",
                       new_x="LMARGIN", new_y="NEXT")

    if s["highlights"]:
        _section(pdf, "Your session")
        for label, value in s["highlights"]:
            _row(pdf, label, value)

    _section(pdf, "What to work on")
    if s["issues"]:
        for desc, how in s["issues"]:
            _row(pdf, desc, how, WARM)
    elif job.get("report"):
        pdf.set_font("Sans", size=10.5)
        pdf.set_text_color(*INK)
        pdf.cell(0, 7.5, "No form problems came up often enough to flag. Well done.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Sans", size=10.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 7.5, "Form feedback is not available for this test yet.", new_x="LMARGIN", new_y="NEXT")

    if s["tips"]:
        _section(pdf, "Tips")
        pdf.set_font("Sans", size=10.5)
        pdf.set_text_color(*INK)
        for tip in s["tips"]:
            pdf.cell(6, 6.5, "•")
            pdf.multi_cell(w - 6, 6.5, tip, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

    filming = FILMING.get(job.get("test_key") or "")
    if filming:
        _section(pdf, "For your next recording")
        pdf.set_font("Sans", size=10.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(w, 6.5, filming + " Use good lighting and wear clothes that contrast with the background.",
                       new_x="LMARGIN", new_y="NEXT")

    pdf.output(out_path)
    return out_path
