"""
GuardianAI - Flask Backend
Agents: Face Verification, Fraud Detection, Risk Scoring, Report Generation, Knowledge Agent
AI: Google Gemini 1.5 Flash (FREE tier)
DB: MongoDB Atlas (FREE tier)
Deploy: Google Cloud Run
"""

import os, json, base64, datetime, re
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER
from dotenv import load_dotenv

# ─── APP ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
load_dotenv()

# ─── CONFIG — set as env vars; fallback strings shown for local testing ────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("DB_NAME", "guardianai")

print("Gemini Key:", "Loaded" if GEMINI_API_KEY else "Missing")
print("MongoDB URI:", "Loaded" if MONGODB_URI else "Missing")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI missing in .env")

# ─── GEMINI ───────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")   # same model handles text + vision

# ─── MONGODB ──────────────────────────────────────────────────────────────────
db = None
fraud_col = risk_col = knowledge_col = face_col = None
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
    client.server_info()
    db           = client[DB_NAME]
    fraud_col    = db["fraud_reports"]
    risk_col     = db["risk_scores"]
    knowledge_col= db["knowledge_base"]
    face_col     = db["face_verifications"]
    print("✅ MongoDB connected")
except Exception as e:
    print(f"⚠️  MongoDB unavailable: {e}  — app runs without DB")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def to_str(obj):
    """Recursively convert ObjectId → str for JSON."""
    if isinstance(obj, dict):  return {k: to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_str(i) for i in obj]
    if isinstance(obj, ObjectId): return str(obj)
    return obj

def mongo_save(col, doc):
    """Insert doc, return inserted id string or None."""
    if col is None: return None
    try:
        doc["created_at"] = datetime.datetime.utcnow().isoformat()
        return str(col.insert_one(doc).inserted_id)
    except Exception as e:
        print(f"Mongo save error: {e}")
        return None

def clean_json(text):
    """Strip markdown fences and parse JSON robustly."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?[\r\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\r\n]*```$", "", text)
    text = text.strip()
    return json.loads(text)

def gemini(prompt, image_parts=None):
    """Call Gemini. image_parts = list of {'mime_type':..,'data':bytes}"""
    if image_parts:
        parts = [prompt] + image_parts
        return model.generate_content(parts).text
    return model.generate_content(prompt).text

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "gemini": "gemini-1.5-flash", "db": db is not None})

# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — FACE VERIFICATION AGENT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/face-verify", methods=["POST"])
def face_verify():
    """
    Input : selfie_b64, id_b64 (base64 strings), doc_type, name
    Output: verdict, confidence, findings, document_authentic, recommendation
    """
    data       = request.get_json(silent=True) or {}
    selfie_b64 = data.get("selfie_b64", "").strip()
    id_b64     = data.get("id_b64",     "").strip()
    doc_type   = data.get("doc_type",   "Government ID")
    name       = data.get("name",       "")

    has_images = bool(selfie_b64 and id_b64)

    prompt = f"""You are GuardianAI Face Verification Agent — an expert in biometric identity verification.

{"TASK: Two images are provided. The FIRST image is the person's SELFIE. The SECOND image is their " + doc_type + (" belonging to " + name if name else "") + ". Compare the faces carefully." if has_images else "TASK: No images provided. Generate a realistic demo verification response."}

Analyse:
1. Facial similarity between selfie and ID photo
2. Whether the ID document looks authentic (not tampered/fake)
3. Image quality and liveness indicators

Respond ONLY in valid JSON (no markdown, no extra text):
{{
  "verdict": "MATCH",
  "confidence": 87,
  "face_similarity_score": 87,
  "findings": [
    "Facial structure matches across both images",
    "Eye spacing and nose bridge are consistent",
    "ID document appears genuine with visible security features"
  ],
  "document_authentic": true,
  "liveness_check": "Pass",
  "recommendation": "Approve",
  "notes": "High confidence match. Safe to proceed."
}}

Use "MATCH", "NO MATCH", or "UNCERTAIN" for verdict.
Use "Approve", "Review", or "Reject" for recommendation."""

    try:
        if has_images:
            img_parts = [
                {"mime_type": "image/jpeg", "data": base64.b64decode(selfie_b64)},
                {"mime_type": "image/jpeg", "data": base64.b64decode(id_b64)},
            ]
            raw = gemini(prompt, img_parts)
        else:
            raw = gemini(prompt)

        result = clean_json(raw)

    except Exception as e:
        print(f"Face verify error: {e}")
        result = {
            "verdict": "UNCERTAIN",
            "confidence": 60,
            "face_similarity_score": 60,
            "findings": [
                "Image analysis could not be completed",
                "Please ensure clear, well-lit photos",
                "Manual review recommended"
            ],
            "document_authentic": None,
            "liveness_check": "Unknown",
            "recommendation": "Review",
            "notes": "Automated analysis unavailable. Manual verification needed."
        }

    # Save to MongoDB
    doc_id = mongo_save(face_col, {
        "doc_type": doc_type,
        "name": name,
        "has_images": has_images,
        "result": result
    })
    result["_id"] = doc_id
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — FRAUD DETECTION AGENT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/fraud-detection", methods=["POST"])
def fraud_detection():
    """
    Input : description (text), category, image_b64 (optional), image_mime
    Output: fraud_probability, verdict, risk_level, red_flags, actions
    """
    data        = request.get_json(silent=True) or {}
    description = data.get("description", "").strip()
    category    = data.get("category", "General Fraud")
    image_b64   = data.get("image_b64", "").strip()
    image_mime  = data.get("image_mime", "image/jpeg")

    if not description and not image_b64:
        return jsonify({"error": "Provide a description or upload an image"}), 400

    prompt = f"""You are GuardianAI Fraud Detection Agent — an expert in identifying online fraud, scams, and financial crimes in India.

Category: {category}
Description: {description if description else "(Analyse the provided image for fraud indicators)"}

{"An image/screenshot has also been provided — analyse it for additional fraud signals." if image_b64 else ""}

Analyse for fraud and respond ONLY in valid JSON (no markdown):
{{
  "fraud_probability": 85,
  "verdict": "Real Fraud",
  "risk_level": "High",
  "fraud_type": "Bank Phishing Scam",
  "red_flags": [
    "Unsolicited contact claiming urgency",
    "Requesting OTP or password",
    "Unofficial communication channel used"
  ],
  "tactics_used": "The fraudster impersonates a bank official and creates urgency to extract OTP.",
  "immediate_actions": [
    "Do NOT share any OTP or password",
    "Block the sender immediately",
    "Report to cybercrime.gov.in"
  ],
  "prevention_tips": [
    "Banks never ask for OTP over phone or WhatsApp",
    "Always verify caller identity via official bank number"
  ],
  "confidence": 92
}}

fraud_probability must be 0-100. verdict must be one of: Real Fraud, Likely Fraud, Suspicious, Appears Legitimate."""

    try:
        if image_b64:
            img_parts = [{"mime_type": image_mime, "data": base64.b64decode(image_b64)}]
            raw = gemini(prompt, img_parts)
        else:
            raw = gemini(prompt)

        result = clean_json(raw)

    except json.JSONDecodeError:
        # Gemini returned text not JSON — extract key values
        prob = 70
        m = re.search(r'"fraud_probability"\s*:\s*(\d+)', raw or "")
        if m: prob = int(m.group(1))
        result = {
            "fraud_probability": prob,
            "verdict": "Likely Fraud" if prob > 50 else "Suspicious",
            "risk_level": "High" if prob > 70 else "Medium",
            "fraud_type": category,
            "red_flags": ["Suspicious activity patterns detected"],
            "tactics_used": "Analysis indicates potential fraudulent activity.",
            "immediate_actions": ["Do not send money", "Contact your bank", "Report to cybercrime.gov.in"],
            "prevention_tips": ["Verify all requests independently", "Never share OTP or passwords"],
            "confidence": 65
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    doc_id = mongo_save(fraud_col, {
        "category": category,
        "description": description[:500],
        "has_image": bool(image_b64),
        "result": result
    })
    result["_id"] = doc_id
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — RISK SCORING AGENT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/risk-score", methods=["POST"])
def risk_score():
    """
    Input : amount, name, contact method, flags[], description
    Output: risk_score (0-100), risk_level, score_breakdown, recommendations
    """
    data    = request.get_json(silent=True) or {}
    amount  = data.get("amount", "")
    name    = data.get("name", "Unknown")
    contact = data.get("contact", "Unknown")
    flags   = data.get("flags", [])
    desc    = data.get("description", "")

    flag_descriptions = {
        "urgency":      "Urgency or pressure to act fast",
        "gift_cards":   "Requesting gift card payment",
        "unknown_link": "Sent unknown or suspicious link",
        "too_good":     "Offer seems too good to be true",
        "secret":       "Asked to keep transaction secret",
        "otp":          "Asked for OTP or password"
    }
    flags_text = ", ".join([flag_descriptions.get(f, f) for f in flags]) if flags else "None"

    prompt = f"""You are GuardianAI Risk Scoring Agent. Calculate a precise numerical fraud risk score.

Signals provided:
- Transaction Amount: ₹{amount if amount else 'Not specified'}
- Other Party Name: {name}
- Contact Method Used: {contact}
- Risk Flags Present: {flags_text}
- Situation Description: {desc if desc else 'Not provided'}

Score each dimension out of 25. Total = risk_score out of 100.

Respond ONLY in valid JSON (no markdown):
{{
  "risk_score": 72,
  "risk_level": "High",
  "score_breakdown": {{
    "amount_risk": 18,
    "contact_risk": 20,
    "flag_risk": 22,
    "context_risk": 12
  }},
  "top_risk_factors": [
    "High-value transaction to unknown party",
    "Contact via unverified WhatsApp channel",
    "Multiple fraud flags detected"
  ],
  "recommendations": [
    "Stop all communication with this party immediately",
    "Verify identity through official channels before proceeding",
    "Report to your bank's fraud helpline"
  ],
  "confidence": 88
}}

risk_level must be: Low (0-30), Medium (31-60), High (61-80), or Critical (81-100)."""

    try:
        raw    = gemini(prompt)
        result = clean_json(raw)
    except Exception:
        base_score = min(100, len(flags) * 14 + 25)
        result = {
            "risk_score": base_score,
            "risk_level": "Critical" if base_score>80 else ("High" if base_score>60 else ("Medium" if base_score>30 else "Low")),
            "score_breakdown": {
                "amount_risk":  15 if amount else 5,
                "contact_risk": 12,
                "flag_risk":    min(25, len(flags) * 8),
                "context_risk": 10
            },
            "top_risk_factors": [flag_descriptions.get(f, f) for f in flags[:3]] or ["Insufficient data for full analysis"],
            "recommendations": ["Do not send money", "Verify identity independently", "Contact bank if money already sent"],
            "confidence": 55
        }

    doc_id = mongo_save(risk_col, {
        "amount": amount, "name": name, "contact": contact,
        "flags": flags, "description": desc[:300],
        "result": result
    })
    result["_id"] = doc_id
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 4 — REPORT GENERATION AGENT  →  returns real PDF
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    """
    Input : case details + optional fraud/risk results
    Output: PDF file download
    """
    data         = request.get_json(silent=True) or {}
    subject_name = data.get("subject_name", "Unknown Subject")
    case_id      = data.get("case_id",      "GRDAI-001")
    inv_type     = data.get("investigation_type", "Fraud Investigation")
    summary      = data.get("summary", "").strip()
    analyst      = data.get("analyst_name", "GuardianAI System")
    fraud_res    = data.get("fraud_result", {})
    risk_res     = data.get("risk_result",  {})

    if not summary:
        return jsonify({"error": "Incident summary is required"}), 400

    # ── Ask Gemini to write the narrative ──
    prompt = f"""You are GuardianAI Report Writer. Write a formal, professional fraud investigation report.

CASE DETAILS:
- Case ID: {case_id}
- Subject: {subject_name}
- Investigation Type: {inv_type}
- Lead Analyst: {analyst}
- Date: {datetime.datetime.now().strftime("%d %B %Y")}

INCIDENT SUMMARY:
{summary}

AI ANALYSIS RESULTS:
- Fraud Probability: {fraud_res.get('fraud_probability', 'N/A')}%
- Fraud Verdict: {fraud_res.get('verdict', 'N/A')}
- Fraud Type: {fraud_res.get('fraud_type', 'N/A')}
- Risk Score: {risk_res.get('risk_score', 'N/A')}/100
- Risk Level: {risk_res.get('risk_level', 'N/A')}
- Red Flags: {', '.join(fraud_res.get('red_flags', [])) or 'N/A'}

Write a complete investigation report. Use EXACTLY these section headers (all caps, on their own line):

EXECUTIVE SUMMARY
INCIDENT DETAILS
FRAUD INDICATORS IDENTIFIED
TECHNICAL ANALYSIS
RISK ASSESSMENT
RECOMMENDED ACTIONS
CASE DISPOSITION
ANALYST NOTES

Each section: 3-5 sentences. Be professional, specific, and factual."""

    try:
        report_text = gemini(prompt)
    except Exception as e:
        report_text = f"EXECUTIVE SUMMARY\nThis report documents a fraud investigation for case {case_id} involving {subject_name}.\n\nINCIDENT DETAILS\n{summary}\n\nRISK ASSESSMENT\nRisk Score: {risk_res.get('risk_score','N/A')}/100. Risk Level: {risk_res.get('risk_level','N/A')}.\n\nRECOMMENDED ACTIONS\nReport to cybercrime.gov.in. Contact your bank immediately. Do not send any further money.\n\nCASE DISPOSITION\nCase is open and under review."

    # ── Build PDF with ReportLab ──
    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.8*inch, leftMargin=0.8*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )
    styles  = getSampleStyleSheet()
    story   = []
    now_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M IST")

    # Header
    hdr = ParagraphStyle("hdr", parent=styles["Heading1"],
                         fontSize=22, textColor=colors.HexColor("#0f172a"),
                         alignment=TA_CENTER, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"],
                         fontSize=10, textColor=colors.HexColor("#64748b"),
                         alignment=TA_CENTER)
    story += [
        Paragraph("🛡️  GuardianAI", hdr),
        Paragraph("FRAUD INVESTIGATION REPORT", hdr),
        Spacer(1, 6),
        Paragraph(f"Generated on {now_str}", sub),
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0ea5e9")),
        Spacer(1, 12),
    ]

    # Metadata table
    fraud_prob = fraud_res.get("fraud_probability", "N/A")
    risk_score_val = risk_res.get("risk_score", "N/A")
    risk_level_val = risk_res.get("risk_level", "N/A")
    verdict_val    = fraud_res.get("verdict", "N/A")

    meta_data = [
        ["Case ID",        case_id,          "Date",         now_str.split(",")[0]],
        ["Subject",        subject_name,     "Analyst",      analyst],
        ["Type",           inv_type,         "Fraud Prob.",  f"{fraud_prob}%"],
        ["Risk Score",     f"{risk_score_val}/100", "Risk Level", risk_level_val],
        ["Verdict",        verdict_val,      "Status",       "Under Investigation"],
    ]
    tbl = Table(meta_data, colWidths=[1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch])
    tbl.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (2,0),(2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("TEXTCOLOR",   (0,0),(0,-1),  colors.HexColor("#475569")),
        ("TEXTCOLOR",   (2,0),(2,-1),  colors.HexColor("#475569")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#f1f5f9"),colors.HexColor("#ffffff")]),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0),(-1,-1), 7),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [tbl, Spacer(1, 20)]

    # Body sections
    sec_style = ParagraphStyle("sec", parent=styles["Heading2"],
                               fontSize=12, textColor=colors.HexColor("#0ea5e9"),
                               spaceBefore=16, spaceAfter=5,
                               borderPad=4)
    body_style = ParagraphStyle("body", parent=styles["Normal"],
                                fontSize=10, textColor=colors.HexColor("#334155"),
                                leading=17, spaceAfter=4)
    bullet_style = ParagraphStyle("bullet", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#334155"),
                                  leading=17, leftIndent=14, spaceAfter=2)

    current_header = None
    buffer_lines   = []

    KNOWN_SECTIONS = {
        "EXECUTIVE SUMMARY", "INCIDENT DETAILS", "FRAUD INDICATORS IDENTIFIED",
        "TECHNICAL ANALYSIS", "RISK ASSESSMENT", "RECOMMENDED ACTIONS",
        "CASE DISPOSITION", "ANALYST NOTES"
    }

    def flush(header, lines):
        if not header: return
        story.append(Paragraph(header, sec_style))
        for ln in lines:
            ln = ln.strip()
            if not ln: continue
            if ln.startswith(("- ", "• ", "* ")):
                story.append(Paragraph("• " + ln[2:], bullet_style))
            else:
                story.append(Paragraph(ln, body_style))

    for line in report_text.splitlines():
        stripped = line.strip()
        is_section = stripped.upper() in KNOWN_SECTIONS or (
            stripped.isupper() and 5 < len(stripped) < 55 and not stripped.startswith("{")
        )
        if is_section:
            flush(current_header, buffer_lines)
            current_header = stripped.upper()
            buffer_lines = []
        elif stripped:
            buffer_lines.append(stripped)

    flush(current_header, buffer_lines)

    # If Gemini returned unstructured text, just dump it
    if not any(isinstance(e, Paragraph) and e.style.name == "sec" for e in story):
        for para in report_text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))

    # Footer
    story += [
        Spacer(1, 24),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 6),
        Paragraph(
            f"GuardianAI v1.0 · Powered by Google Gemini · Case {case_id} · {now_str}",
            ParagraphStyle("ft", parent=styles["Normal"], fontSize=7.5,
                           textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER)
        )
    ]

    doc.build(story)
    buffer.seek(0)

    # Save record to MongoDB
    mongo_save(fraud_col, {
        "case_id": case_id, "subject_name": subject_name,
        "investigation_type": inv_type, "analyst": analyst,
        "summary": summary[:500], "pdf_generated": True,
        "fraud_result": fraud_res, "risk_result": risk_res
    })

    return send_file(
        buffer, mimetype="application/pdf",
        as_attachment=True,
        download_name=f"GuardianAI_{case_id}.pdf"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 5 — KNOWLEDGE AGENT  (MongoDB search + stats)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/knowledge/search", methods=["POST"])
def knowledge_search():
    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Use Gemini to extract keywords
    try:
        raw      = gemini(f'Extract 3-5 search keywords from: "{query}"\nReturn ONLY a JSON array like: ["word1","word2","word3"]')
        raw      = re.sub(r"^```(?:json)?", "", raw.strip()).rstrip("` \n")
        keywords = json.loads(raw)
        if not isinstance(keywords, list): raise ValueError
    except Exception:
        keywords = [w for w in query.split() if len(w) > 3][:5] or [query]

    results = []
    if fraud_col is not None:
        try:
            pattern = "|".join(re.escape(k) for k in keywords)
            cursor  = fraud_col.find({"$or": [
                {"description":       {"$regex": pattern, "$options": "i"}},
                {"category":          {"$regex": pattern, "$options": "i"}},
                {"result.fraud_type": {"$regex": pattern, "$options": "i"}},
                {"result.verdict":    {"$regex": pattern, "$options": "i"}},
            ]}).sort("created_at", -1).limit(6)
            results = to_str(list(cursor))
        except Exception as e:
            print(f"Knowledge search error: {e}")

    return jsonify({"keywords": keywords, "results": results, "count": len(results)})


@app.route("/api/knowledge/stats", methods=["GET"])
def knowledge_stats():
    stats = {"total_reports": 0, "total_risk_scores": 0,
             "high_risk_count": 0, "top_fraud_types": [], "db_connected": db is not None}
    if fraud_col is not None:
        try:
            stats["total_reports"]     = fraud_col.count_documents({})
            stats["total_risk_scores"] = risk_col.count_documents({}) if risk_col else 0
            stats["high_risk_count"]   = fraud_col.count_documents(
                {"result.risk_level": {"$in": ["High", "Critical"]}})
            pipeline = [
                {"$group": {"_id": "$result.fraud_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}, {"$limit": 5}
            ]
            stats["top_fraud_types"] = [
                {"type": d["_id"], "count": d["count"]}
                for d in fraud_col.aggregate(pipeline) if d["_id"]
            ]
        except Exception as e:
            print(f"Stats error: {e}")
    return jsonify(stats)


@app.route("/api/knowledge/save", methods=["POST"])
def knowledge_save():
    data = request.get_json(silent=True) or {}
    if not data.get("description"):
        return jsonify({"error": "description required"}), 400
    doc_id = mongo_save(knowledge_col, data)
    return jsonify({"saved": True, "_id": doc_id})


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False
    )
