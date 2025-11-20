from flask import Flask, request, render_template, send_from_directory
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os

app = Flask(__name__, template_folder='../templates', static_folder='../public')

# تحميل النموذج مرة واحدة فقط
print("جاري تحميل النموذج...")
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
print("تم تحميل النموذج بنجاح!")

# قائمة الأعراض
SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"},
    {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"},
    {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"},
    {"key": "dizziness", "text": "دوخة"},
    {"key": "Voice quality changes", "text": "تغيرات في جودة الصوت"},
    {"key": "Hoarseness", "text": "بحة الصوت"},
    {"key": "Taste changes", "text": "تغير الطعم"},
    {"key": "Decreased appetite", "text": "انخفاض الشهية"},
    {"key": "Vomiting", "text": "تقيؤ"},
    {"key": "Heartburn", "text": "حرقة صدر"},
    {"key": "Gas", "text": "الغازات"},
    {"key": "Bloating", "text": "الانتفاخ"},
    {"key": "Hiccups", "text": "زغطة"},
    {"key": "Constipation", "text": "امساك"},
    {"key": "Diarrhea", "text": "اسهال"},
    {"key": "Fecal incontinence", "text": "سلس برازي"},
    {"key": "Shortness of breath", "text": "ضيق تنفس"},
]

symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

# الأسئلة التفصيلية (كل عرض بأسئلته)
SYMPTOM_QUESTIONS = {
    "dry_mouth": [{"question": "في الأيام السبعة الماضية، ما شدة جفاف الفم؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]}],
    "headache": [{"question": "في الأيام السبعة الماضية، ما شدة الصداع؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]}],
    "nausea": [
        {"question": "في الأيام السبعة الماضية، ما شدة الغثيان؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]},
        {"question": "هل أثر الغثيان على الأكل أو الشرب؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]}
    ],
    "abdominal_pain": [
        {"question": "في الأيام السبعة الماضية، ما شدة ألم البطن؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]},
        {"question": "هل أثر الألم على الأكل أو الشرب؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]},
        {"question": "هل أثر الألم على نشاطك اليومي؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]}
    ],
    "cough": [
        {"question": "في الأيام السبعة الماضية، ما شدة السعال؟", "options": ["لا أبدا","قليل","متوسط","شديد","شديد جدًا"]},
        {"question": "هل السعال جاف أم فيه بلغم؟", "options": ["جاف", "فيه بلغم", "متقلب"]}
    ],
}

# كشف الأعراض
def detect_symptoms(text, threshold=0.28):
    parts = re.split(r"[،,.\n!؟؛]", text)
    detected = set()
    def cos(a, b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
    for part in parts:
        part = part.strip()
        if len(part) < 3: continue
        emb = model.encode(part)
        for i, sim in enumerate([cos(emb, e) for e in symptom_embeddings]):
            if sim > threshold:
                detected.add(SYMPTOMS[i]["key"])
    return list(detected)

# جلسة المستخدم (مؤقتة)
session = {"chats": [], "pending": [], "answers": {}, "completed": False, "questions": {}}

@app.route("/", methods=["GET", "POST"])
def index():
    draft = ""
    current_question = ""
    options = []

    if request.method == "POST":
        if "answer" in request.form:
            if session["pending"]:
                sym = session["pending"][0]
                answer = request.form["answer"]
                session["answers"][sym] = session["answers"].get(sym, []) + [answer]
                session["chats"].append((session["questions"][sym].pop(0)["question"], answer))
                if not session["questions"][sym]:
                    session["pending"].pop(0)
                if not session["pending"]:
                    session["completed"] = True
        else:
            text = request.form.get("symptoms", "").strip()
            if text:
                session["chats"].append((text, ""))
                detected = detect_symptoms(text)
                session["pending"] = [d for d in detected if d in SYMPTOM_QUESTIONS]
                session["questions"] = {k: [q.copy() for q in v] for k, v in SYMPTOM_QUESTIONS.items() if k in session["pending"]}
                if session["pending"]:
                    session["chats"][-1] = (text, f"تمام، لقيت {len(session['pending'])} عرض، هسألك عليهم")
                else:
                    session["chats"][-1] = (text, "مش لاقي أعراض واضحة، ممكن توضح أكتر؟")
                draft = text

    if session["pending"]:
        current_sym = session["pending"][0]
        current_question = session["questions"][current_sym][0]["question"]
        options = session["questions"][current_sym][0]["options"]

    return render_template("index.html",
        chats=session["chats"],
        pending=session["pending"],
        completed=session["completed"],
        current_question=current_question,
        options=options,
        draft=draft
    )

@app.route('/style.css')
def style():
    return send_from_directory('../public', 'style.css')

if __name__ == "__main__":
    app.run(debug=True)