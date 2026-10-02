"""Resume Screening AI - Flask web app.

Run:  python app.py   then open http://127.0.0.1:5000
"""
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from screener import ALLOWED_EXTENSIONS, extract_text, rank_resumes

BASE_DIR = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "sample_data"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB per request
MAX_FILES = 100


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/rank")
def api_rank():
    job_description = (request.form.get("job_description") or "").strip()
    files = [f for f in request.files.getlist("resumes") if f.filename]

    if len(job_description.split()) < 5:
        return jsonify(error="Paste a job description of at least a few words."), 400
    if not files:
        return jsonify(error="Upload at least one resume (PDF, DOCX or TXT)."), 400
    if len(files) > MAX_FILES:
        return jsonify(error=f"Upload at most {MAX_FILES} resumes at a time."), 400

    resumes, skipped = [], []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append({"filename": f.filename, "reason": "Unsupported file type"})
            continue
        try:
            text = extract_text(f.filename, f.read()).strip()
        except Exception:
            skipped.append({"filename": f.filename, "reason": "File could not be read"})
            continue
        if len(text.split()) < 20:
            skipped.append({"filename": f.filename, "reason": "No readable text (scanned image?)"})
            continue
        resumes.append({"filename": f.filename, "text": text})

    if not resumes:
        return jsonify(error="None of the uploaded files had readable text.", skipped=skipped), 400
    try:
        results = rank_resumes(job_description, resumes)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(results=results, skipped=skipped)


@app.get("/api/sample")
def api_sample():
    """Rank the bundled sample resumes against the bundled sample job description."""
    jd = (SAMPLE_DIR / "job_description.txt").read_text(encoding="utf-8")
    resumes = [{"filename": p.name, "text": p.read_text(encoding="utf-8")}
               for p in sorted((SAMPLE_DIR / "resumes").glob("*.txt"))]
    return jsonify(job_description=jd, results=rank_resumes(jd, resumes), skipped=[])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
