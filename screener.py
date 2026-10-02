"""NLP engine: parses resumes and ranks them against a job description.

Final score (0-100) is a weighted blend of:
  * semantic similarity - TF-IDF (1-2 grams) cosine similarity between resume and job description
  * skill coverage      - share of the job description's skills that appear in the resume
  * experience fit      - years of experience in the resume vs. years asked for in the job description
Components that cannot be measured (e.g. no skills detected in the job description) are dropped
and the remaining weights are re-normalised.
"""
import io
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills import SKILLS

WEIGHTS = {"semantic": 0.50, "skills": 0.35, "experience": 0.15}
SEMANTIC_CEILING = 0.50  # a raw cosine of 0.50 or more counts as a full semantic match
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
TOKEN_RE = re.compile(r"[a-z][a-z0-9+#]*(?:\.[a-z0-9]+)*")
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)", re.I)

EXTRA_STOP = {"experience", "work", "working", "year", "years", "team", "strong", "good", "ability",
              "skills", "skill", "knowledge", "etc", "including", "will", "role", "candidate",
              "responsible", "responsibilities", "looking", "required", "preferred", "must"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP


# ---------------------------------------------------------------- text extraction
def extract_text(filename: str, data: bytes) -> str:
    """Return plain text from a .pdf, .docx or .txt file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        return "\n".join(parts)
    if ext == ".txt":
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
    raise ValueError(f"Unsupported file type: {ext or 'unknown'}")


# ---------------------------------------------------------------- preprocessing
def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def stem(word: str) -> str:
    """Very small suffix stemmer so 'managed/managing/manages' match 'manage'."""
    if len(word) <= 4:
        return word
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                return word
            return word[: -len(suffix)]
    return word


def tokenize(text: str):
    tokens = []
    for tok in TOKEN_RE.findall(text):
        if tok in STOP_WORDS or len(tok) < 2 and tok not in {"r", "c"}:
            continue
        tokens.append(stem(tok))
    return tokens


def surface_map(*texts):
    """Map each stemmed token back to a readable word (first form seen, job description first)."""
    forms = {}
    for text in texts:
        for tok in TOKEN_RE.findall(text):
            if tok not in STOP_WORDS:
                forms.setdefault(stem(tok), tok)
    return forms


def readable(term: str, forms: dict) -> str:
    return " ".join(forms.get(part, part) for part in term.split())


# ---------------------------------------------------------------- feature extraction
def _skill_pattern(alias: str):
    return re.compile(r"(?<![a-z0-9+#])" + re.escape(alias) + r"(?![a-z0-9+#])")


_SKILL_PATTERNS = {name: [_skill_pattern(a) for a in aliases] for name, aliases in SKILLS.items()}


def find_skills(text: str):
    lowered = text.lower()
    return {name for name, pats in _SKILL_PATTERNS.items() if any(p.search(lowered) for p in pats)}


def find_years(text: str, cap: int = 40):
    """Largest 'N years' value mentioned (ignores impossible values)."""
    values = [int(m.group(1)) for m in YEARS_RE.finditer(text)]
    values = [v for v in values if 0 < v <= cap]
    return max(values) if values else None


def guess_name(filename: str, text: str) -> str:
    stem_name = re.sub(r"[_\-.]+", " ", Path(filename).stem)
    stem_name = re.sub(r"(?i)\b(resume|cv|curriculum vitae)\b", "", stem_name).strip()
    return stem_name.title() if stem_name else filename


# ---------------------------------------------------------------- ranking
def rank_resumes(job_description: str, resumes):
    """resumes: list of dicts {"filename": str, "text": str}. Returns ranked list of result dicts."""
    jd_clean = clean_text(job_description)
    if not tokenize(jd_clean):
        raise ValueError("The job description has no usable text.")

    jd_skills = find_skills(job_description)
    jd_years = find_years(job_description, cap=25)

    cleaned = [clean_text(r["text"]) for r in resumes]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([jd_clean] + cleaned)
    except ValueError:  # empty vocabulary
        matrix = None

    terms = vectorizer.get_feature_names_out() if matrix is not None else []
    forms = surface_map(jd_clean, *cleaned)
    results = []
    for i, resume in enumerate(resumes):
        text = resume["text"]
        raw_cos = float(cosine_similarity(matrix[0], matrix[i + 1])[0][0]) if matrix is not None else 0.0
        semantic = min(1.0, raw_cos / SEMANTIC_CEILING)

        resume_skills = find_skills(text)
        matched = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)
        extra = sorted(resume_skills - jd_skills)
        skill_cov = len(matched) / len(jd_skills) if jd_skills else None

        years = find_years(text)
        exp_fit = None
        if jd_years:
            exp_fit = min(1.0, (years or 0) / jd_years)

        parts = {"semantic": semantic, "skills": skill_cov, "experience": exp_fit}
        active = {k: v for k, v in parts.items() if v is not None}
        total_w = sum(WEIGHTS[k] for k in active)
        final = sum(WEIGHTS[k] * v for k, v in active.items()) / total_w

        # keywords that drive the match: high weight in both the resume and the job description
        keywords = []
        if matrix is not None:
            overlap = matrix[0].multiply(matrix[i + 1]).tocoo()
            top = sorted(zip(overlap.data, overlap.col), reverse=True)[:30]
            for _, col in top:
                term = readable(terms[col], forms)
                if not any(term in k or k in term for k in keywords):
                    keywords.append(term)
                if len(keywords) == 8:
                    break

        results.append({
            "filename": resume["filename"],
            "name": guess_name(resume["filename"], text),
            "email": (EMAIL_RE.search(text).group(0) if EMAIL_RE.search(text) else ""),
            "score": round(final * 100, 1),
            "semantic": round(semantic * 100, 1),
            "skill_coverage": None if skill_cov is None else round(skill_cov * 100, 1),
            "experience_fit": None if exp_fit is None else round(exp_fit * 100, 1),
            "years": years,
            "required_years": jd_years,
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills": extra[:10],
            "keywords": keywords,
            "word_count": len(text.split()),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results
