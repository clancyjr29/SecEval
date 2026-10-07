"""SecEval - review workbench for AI-generated cybersecurity content.

Run:  pip install -r requirements.txt && uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs (interactive UI).
Flow: POST /users -> copy your API key -> click "Authorize" in /docs.

This app only stores and analyzes text. It never executes submitted code.
"""
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

DB = os.environ.get("SECEVAL_DB", "seceval.db")  # point at a persistent disk when hosting
SIGNUP_CODE = os.environ.get("SECEVAL_SIGNUP_CODE")  # set this on any public deployment
DIMS = ["accuracy", "technical_validity", "completeness", "clarity", "responsibility"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, key_hash TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY, owner INTEGER, kind TEXT, category TEXT,
  prompt TEXT, body TEXT, meta TEXT, created REAL);
CREATE TABLE IF NOT EXISTS reviews(
  id INTEGER PRIMARY KEY, item_id INTEGER, reviewer INTEGER,
  scores TEXT, claims TEXT, verdict TEXT, feedback TEXT, created REAL);
"""

app = FastAPI(title="SecEval", description="Review AI-generated security content for accuracy.")
with closing(sqlite3.connect(DB)) as _c:
    _c.executescript(SCHEMA)


# ---------- infra ----------
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


api_key_header = APIKeyHeader(name="X-API-Key")


def current_user(x_api_key: str = Depends(api_key_header), con=Depends(db)):
    h = hashlib.sha256(x_api_key.encode()).hexdigest()
    row = con.execute("SELECT * FROM users WHERE key_hash=?", (h,)).fetchone()
    if not row:
        raise HTTPException(401, "Invalid API key")
    return row


# ---------- automatic fact checks ----------
CVE_RE = re.compile(r"\bCVE-(\d{4})-\d{4,}\b", re.I)
ATTACK_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
CVSS_RE = re.compile(
    r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)
W = {"AV": {"N": .85, "A": .62, "L": .55, "P": .2}, "AC": {"L": .77, "H": .44},
     "UI": {"N": .85, "R": .62}, "CIA": {"H": .56, "L": .22, "N": 0}}
PR = {"U": {"N": .85, "L": .62, "H": .27}, "C": {"N": .85, "L": .68, "H": .5}}


def roundup(x: float) -> float:
    i = round(x * 100000)
    return i / 100000.0 if i % 10000 == 0 else (math.floor(i / 10000) + 1) / 10.0


def cvss3_score(vec: str) -> float:
    m = dict(p.split(":") for p in vec.split("/")[1:])
    s = m["S"]
    iss = 1 - (1 - W["CIA"][m["C"]]) * (1 - W["CIA"][m["I"]]) * (1 - W["CIA"][m["A"]])
    imp = 6.42 * iss if s == "U" else 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    ex = 8.22 * W["AV"][m["AV"]] * W["AC"][m["AC"]] * PR[s][m["PR"]] * W["UI"][m["UI"]]
    if imp <= 0:
        return 0.0
    total = imp + ex if s == "U" else 1.08 * (imp + ex)
    return roundup(min(total, 10))


def auto_checks(text: str) -> dict:
    year = time.gmtime().tm_year
    cves = sorted({m.group(0).upper(): int(m.group(1)) for m in CVE_RE.finditer(text)}.items())
    out = {
        "cves": [{"id": c, "plausible_year": 1999 <= y <= year} for c, y in cves],
        "attack_ids": sorted(set(ATTACK_RE.findall(text))),
        "cvss": [],
        "notes": ["CVE/ATT&CK IDs are format-checked only. Verify each against NVD and "
                  "attack.mitre.org. CVSS 'claimed' is a heuristic (first x.y number after the vector)."],
    }
    for m in CVSS_RE.finditer(text):
        calc = cvss3_score(m.group())
        near = re.search(r"\b\d{1,2}\.\d\b", text[m.end(): m.end() + 80])
        claimed = float(near.group()) if near else None
        out["cvss"].append({"vector": m.group(), "computed": calc, "claimed": claimed,
                            "mismatch": claimed is not None and abs(claimed - calc) > 0.05})
    return out


# ---------- schemas ----------
class UserIn(BaseModel):
    name: str
    signup_code: str = ""


class ItemIn(BaseModel):
    kind: str = Field(pattern="^(ai_output|problem)$",
                      description="ai_output = AI answer to evaluate; problem = a problem you designed")
    category: str = Field(description="e.g. threat-analysis, vuln-assessment, offensive-technique, detection")
    prompt: str = Field(description="Prompt given to the AI, or the problem statement")
    body: str = Field(description="The AI's answer, or your reference solution/explanation")
    meta: dict = Field(default_factory=dict, description="e.g. difficulty, grading criteria, model name")


class Claim(BaseModel):
    text: str
    verdict: str = Field(pattern="^(correct|incorrect|unverifiable)$")
    evidence: str = ""  # CVE, ATT&CK ID, advisory URL, RFC, your own test notes


class ReviewIn(BaseModel):
    scores: dict[str, int]
    claims: list[Claim] = []
    verdict: str = Field(pattern="^(accept|revise|reject)$")
    feedback: str

    @field_validator("scores")
    @classmethod
    def check_scores(cls, v):
        if set(v) != set(DIMS) or not all(1 <= s <= 5 for s in v.values()):
            raise ValueError(f"scores must have exactly {DIMS}, each 1-5")
        return v


# ---------- routes ----------
@app.post("/users", summary="Create a user; the API key is shown once")
def create_user(u: UserIn, con=Depends(db)):
    if SIGNUP_CODE and not secrets.compare_digest(u.signup_code, SIGNUP_CODE):
        raise HTTPException(403, "Invalid signup code")
    key = secrets.token_urlsafe(32)
    con.execute("INSERT INTO users(name,key_hash) VALUES(?,?)",
                (u.name, hashlib.sha256(key.encode()).hexdigest()))
    con.commit()
    return {"name": u.name, "api_key": key}


@app.post("/items", summary="Submit an AI output to review, or a problem you designed")
def create_item(i: ItemIn, user=Depends(current_user), con=Depends(db)):
    cur = con.execute(
        "INSERT INTO items(owner,kind,category,prompt,body,meta,created) VALUES(?,?,?,?,?,?,?)",
        (user["id"], i.kind, i.category, i.prompt, i.body, json.dumps(i.meta), time.time()))
    con.commit()
    return {"id": cur.lastrowid, "auto_checks": auto_checks(i.body)}


@app.get("/items", summary="List items (shared workspace)")
def list_items(kind: str | None = None, category: str | None = None,
               user=Depends(current_user), con=Depends(db)):
    q, args = "SELECT id,kind,category,substr(prompt,1,120) AS prompt FROM items WHERE 1=1", []
    for col, val in (("kind", kind), ("category", category)):
        if val:
            q += f" AND {col}=?"
            args.append(val)
    return [dict(r) for r in con.execute(q + " ORDER BY id DESC", args)]


def get_item(con, item_id: int):
    row = con.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Item not found")
    return row


@app.get("/items/{item_id}", summary="Item plus automatic fact checks")
def read_item(item_id: int, user=Depends(current_user), con=Depends(db)):
    r = get_item(con, item_id)
    return {**dict(r), "meta": json.loads(r["meta"]), "auto_checks": auto_checks(r["body"])}


@app.post("/items/{item_id}/reviews", summary="Submit a rubric review with claim-level verdicts")
def add_review(item_id: int, rv: ReviewIn, user=Depends(current_user), con=Depends(db)):
    get_item(con, item_id)
    con.execute(
        "INSERT INTO reviews(item_id,reviewer,scores,claims,verdict,feedback,created) VALUES(?,?,?,?,?,?,?)",
        (item_id, user["id"], json.dumps(rv.scores), json.dumps([c.model_dump() for c in rv.claims]),
         rv.verdict, rv.feedback, time.time()))
    con.commit()
    return {"ok": True}


@app.get("/items/{item_id}/report", response_class=PlainTextResponse,
         summary="Markdown feedback report you can paste into a review platform")
def report(item_id: int, user=Depends(current_user), con=Depends(db)):
    item = get_item(con, item_id)
    rows = con.execute("SELECT r.*, u.name FROM reviews r JOIN users u ON u.id=r.reviewer "
                       "WHERE item_id=? ORDER BY r.id", (item_id,)).fetchall()
    lines = [f"# Review report: item {item_id} ({item['category']})", ""]
    for r in rows:
        scores = json.loads(r["scores"])
        lines += [f"## {r['name']}: {r['verdict'].upper()}", "",
                  "Scores: " + ", ".join(f"{k} {v}/5" for k, v in scores.items()), ""]
        for c in json.loads(r["claims"]):
            ev = f" (evidence: {c['evidence']})" if c["evidence"] else ""
            lines.append(f"- [{c['verdict']}] {c['text']}{ev}")
        lines += ["", r["feedback"], ""]
    return "\n".join(lines)
