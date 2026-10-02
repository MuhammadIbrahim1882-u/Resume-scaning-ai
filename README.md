# Resume Screening AI

Rank resumes against a job description using NLP. Upload PDF, DOCX or TXT resumes, paste a job
description, and get a scored ranking with an explanation for each candidate.

## How it works

Each resume gets a score from 0 to 100, built from three parts:

| Part | Weight | What it measures |
|------|--------|------------------|
| Text match | 50% | TF-IDF (1-2 word phrases) cosine similarity between the resume and the job description |
| Skill coverage | 35% | Share of the job's skills (from a built-in dictionary of ~110 skills) found in the resume |
| Experience fit | 15% | Years of experience in the resume compared with the years the job asks for |

If a part cannot be measured (for example the job lists no years of experience), it is left out
and the remaining weights are rescaled.

NLP steps: lowercasing, removing emails, phone numbers and links, tokenising, stop-word removal,
light stemming, TF-IDF with bigrams and sublinear term frequency.

## Folder structure

```
resume-screening-ai/
  app.py               Flask web server and API
  screener.py          NLP engine (parsing, TF-IDF, scoring)
  skills.py            Skill dictionary - add your own skills here
  requirements.txt
  run.bat / run.sh     One-click setup and start
  index.html           The whole interface in ONE file (styles, script and demo data inside)
  build_demo.py        Refreshes the demo data inside index.html
  sample_data/         Example job description and 4 resumes
  tests/test_screener.py
```

## Run it

Extract the zip first (right-click, Extract All). Do not run files from inside the zip window.

Requires Python 3.9 or newer.

**Windows:** double-click `run.bat`

**Mac / Linux:** `bash run.sh`

**Manually:**

```
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser. (Double-clicking `index.html` shows the full design and the sample data only; to rank your own resumes the server must be running.) Click **Try sample data** to see it working at once.

## Run the tests

```
python -m unittest discover tests
```

## Limits

- Scanned (image-only) PDFs have no text and are skipped. They need OCR first.
- Skill matching uses the dictionary in `skills.py`; skills not listed there are still covered by the text match.
- Scores help you decide whom to read first. They should not be the only basis for a hiring decision.
