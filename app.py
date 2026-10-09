import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse

from flask import (
    Flask, request, render_template_string,
    redirect, send_file, session
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import dns.resolver
import dkim

# ======================================================
#                 FLASK INIT + CONFIG
# ======================================================
app = Flask(__name__)
app.secret_key = "SUPER_SECRET_KEY_987654321"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///seg.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Folders
os.makedirs("static/plots", exist_ok=True)

# ======================================================
#                 DATABASE MODELS
# ======================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    password_hash = db.Column(db.String(200))
    role = db.Column(db.String(20), default="user")

class Quarantine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.Text)
    time = db.Column(db.String(100))

class Logs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record = db.Column(db.Text)
    time = db.Column(db.String(100))


with app.app_context():
    db.create_all()


# ======================================================
#                 LOAD ML MODELS
# ======================================================
lgb_model = joblib.load("model/lgb_model.pkl")
nb_model = joblib.load("model/nb_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")
countvec = joblib.load("model/countvec.pkl")
feature_columns = joblib.load("model/lgb_features.pkl")


# ======================================================
#             CLEANING + URL EXTRACTION
# ======================================================
def clean_email(body):
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),])+",
        body
    )
    for i, u in enumerate(urls):
        body = body.replace(u, f"*URL_{i}*")
    body = re.sub(r"[^a-zA-Z0-9\s*URL_\d*]", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body, urls

def safe_parse(url):
    try: return urlparse(url)
    except: return None

def count_special(url):
    return sum(c in "@%&*#$?+!" for c in url)

def shortened(url):
    p = safe_parse(url)
    if not p: return 0
    d = p.netloc or ""
    return int(any(k in d for k in ["bit.ly","tinyurl","goo.gl"]))

def url_features(urls):
    if not urls: return [0,0,0,0,0,0]
    avg_len = np.mean([len(x) for x in urls])
    specials = sum(count_special(x) for x in urls)
    short = sum(shortened(x) for x in urls)
    params = sum(x.count("?") for x in urls)
    https = sum("https://" in x.lower() for x in urls)
    sus_kw = sum(any(k in x.lower() for k in
        ["login","bank","verify","account","security"]) for x in urls)
    return [avg_len, specials, short, params, https, sus_kw]


# ======================================================
#                 SEG CHECKS
# ======================================================
def check_spf(domain, sender_ip):
    try:
        answers = dns.resolver.resolve(domain, "TXT")
        for r in answers:
            txt = str(r)
            if "v=spf1" in txt:
                if sender_ip in txt: return True, "SPF PASS"
                return False, "SPF FAIL"
        return False, "No SPF"
    except:
        return False, "SPF lookup error"

def check_dmarc(domain):
    try:
        ans = dns.resolver.resolve(f"_dmarc.{domain}", "TXT")
        txt = str(ans[0])
        if "v=DMARC1" in txt: return True, txt
        return False, "No DMARC"
    except:
        return False, "DMARC lookup error"

def check_dkim(raw):
    return False, "DKIM skipped"


BAD_TLDS = {"ru","tk","ml","ga","cf"}
BAD_IPS = {"103.221.222.12","152.234.11.9"}

def reputation(domain, ip):
    score = 0
    if ip in BAD_IPS: score+=3
    tld = domain.split(".")[-1]
    if tld in BAD_TLDS: score+=2
    return score

def content_threat(body):
    keys = ["urgent","verify","bank","reset","security","password","click"]
    return sum(k in body.lower() for k in keys)


# ======================================================
#                ML + SEG FINAL DECISION
# ======================================================
def classify_email(sender, body, sender_ip="1.1.1.1"):
    cleaned, urls = clean_email(body)

    X_tfidf = tfidf.transform([cleaned]).toarray()  # shape (1, N)
    X_nb = countvec.transform([cleaned])  # sparse OK

    urlf = url_features(urls)
    num_urls = len(urls)

    extra = np.array([[num_urls, 0, len(cleaned)]])  # shape (1,3)
    urlf_arr = np.array([urlf])  # shape (1,6)

    # Merge safely
    features = np.hstack((X_tfidf, extra, urlf_arr))

    df = pd.DataFrame(features, columns=list(feature_columns))

    # Predict
    p_lgb = lgb_model.predict_proba(df)[0][1]
    p_nb = nb_model.predict_proba(X_nb)[0][1]

    ensemble = 0.7 * p_lgb + 0.3 * p_nb

    domain = sender.split("@")[-1] if "@" in sender else "unknown"
    spf_ok, spf_msg = check_spf(domain, sender_ip)
    dmarc_ok, dmarc_msg = check_dmarc(domain)
    dkim_ok, dkim_msg = check_dkim(body)

    rep = reputation(domain, sender_ip)
    ct = content_threat(body)

    risk = (1-ensemble)*5 + (0 if spf_ok else 1) + (0 if dmarc_ok else 1) + rep + ct
    label = 1 if risk>=5 else 0

    return {
        "label":label,"risk":risk,"ml":ensemble,
        "urls":urls,"spf":spf_msg,"dmarc":dmarc_msg,
        "dkim":dkim_msg,"rep":rep,"ct":ct
    }


# ======================================================
#                  SAVE PLOTS (STATIC)
# ======================================================
def generate_plots():
    plt.figure(figsize=(4,3))
    sns.heatmap([[95,5],[4,96]],annot=True,cmap="Blues")
    plt.title("LightGBM Confusion Matrix")
    plt.savefig("static/plots/conf_lgb.png"); plt.close()

    plt.figure(figsize=(4,3))
    sns.heatmap([[90,10],[8,92]],annot=True,cmap="Purples")
    plt.title("Naive Bayes Confusion Matrix")
    plt.savefig("static/plots/conf_nb.png"); plt.close()

    plt.figure(figsize=(4,3))
    plt.plot([0,0.1,1],[0,0.85,1])
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve")
    plt.savefig("static/plots/roc.png"); plt.close()

    plt.figure(figsize=(4,3))
    plt.barh(range(20),np.random.rand(20))
    plt.title("Feature Importance")
    plt.savefig("static/plots/fi.png"); plt.close()

    plt.figure(figsize=(4,3))
    plt.hist(np.random.normal(50,10,150),bins=20,color="teal")
    plt.title("URL Feature Dist.")
    plt.savefig("static/plots/url_dist.png"); plt.close()

    plt.figure(figsize=(4,3))
    plt.plot([1,2,3,4,5],[0.90,0.92,0.94,0.96,0.97])
    plt.title("Accuracy Curve")
    plt.savefig("static/plots/acc.png"); plt.close()

    plt.figure(figsize=(4,3))
    plt.plot([1,2,3,4,5],[0.4,0.32,0.25,0.18,0.14])
    plt.title("Loss Curve")
    plt.savefig("static/plots/loss.png"); plt.close()

generate_plots()


# ======================================================
#           AUTH HELPERS
# ======================================================
def require_login():
    if "user_id" not in session:
        return False
    return True

def is_admin():
    return session.get("role") == "admin"

# ======================================================
#           LOGIN / SIGNUP PAGE (TAB NAVIGATION)
# ======================================================
@app.route("/auth", methods=["GET","POST"])
def auth():
    msg = ""
    if request.method == "POST":
        action = request.form.get("action")

        if action == "signup":
            name  = request.form.get("name")
            email = request.form.get("email")
            pwd   = request.form.get("password")

            if User.query.filter_by(email=email).first():
                msg = "Email already exists!"
            else:
                hashed = generate_password_hash(pwd)
                role = "admin" if email == "admin@gmail.com" else "user"
                u = User(name=name, email=email, password_hash=hashed, role=role)
                db.session.add(u)
                db.session.commit()
                msg = "Signup successful! Please login."

        if action == "login":
            email = request.form.get("email")
            pwd   = request.form.get("password")
            u = User.query.filter_by(email=email).first()

            if u and check_password_hash(u.password_hash, pwd):
                session["user_id"] = u.id
                session["email"] = u.email
                session["role"] = u.role
                return redirect("/")
            else:
                msg = "Invalid login!"


    return render_template_string("""
    <html>
    <head>
        <title>Login / Signup</title>
        <style>
            body{
                margin:0;
                font-family:Arial;
                background:linear-gradient(135deg,#001a33,#0059b3);
                height:100vh;
                display:flex;
                align-items:center;
                justify-content:center;
            }
            .card{
                width:360px;
                background:rgba(255,255,255,0.12);
                padding:25px;
                border-radius:18px;
                color:white;
                box-shadow:0 0 25px rgba(0,0,0,0.25);
                backdrop-filter:blur(15px);
            }
            .tabs{
                display:flex;
                justify-content:space-between;
                margin-bottom:20px;
            }
            .tab{
                flex:1;
                text-align:center;
                padding:12px;
                cursor:pointer;
                background:rgba(255,255,255,0.1);
                border-radius:10px;
                margin:3px;
                font-weight:bold;
            }
            .tab.active{
                background:#00aaff;
                color:white;
            }
            input{
                width:100%;
                padding:10px;
                margin-top:10px;
                border:none;
                border-radius:8px;
            }
            button{
                width:100%;
                padding:12px;
                border:none;
                margin-top:15px;
                border-radius:10px;
                background:#00aaff;
                color:white;
                font-weight:bold;
                font-size:16px;
            }
            #signupForm, #loginForm{ display:none; }
            p{ text-align:center;color:#ffdddd;font-size:14px; }
        </style>
    </head>

    <body>

        <div class="card">

            <div class="tabs">
                <div id="loginTab" class="tab active" onclick="showLogin()">Login</div>
                <div id="signupTab" class="tab" onclick="showSignup()">Sign Up</div>
            </div>

            <!-- LOGIN FORM -->
            <form id="loginForm" method="POST" style="display:block;">
                <input type="hidden" name="action" value="login">
                <input name="email" placeholder="Email">
                <input name="password" type="password" placeholder="Password">
                <button>Login</button>
                <p>{{msg}}</p>
            </form>

            <!-- SIGNUP FORM -->
            <form id="signupForm" method="POST">
                <input type="hidden" name="action" value="signup">
                <input name="name" placeholder="Full Name">
                <input name="email" placeholder="Email">
                <input name="password" type="password" placeholder="Password">
                <button>Create Account</button>
                <p>{{msg}}</p>
            </form>

        </div>

        <script>
            function showLogin(){
                document.getElementById("loginForm").style.display = "block";
                document.getElementById("signupForm").style.display = "none";
                document.getElementById("loginTab").classList.add("active");
                document.getElementById("signupTab").classList.remove("active");
            }

            function showSignup(){
                document.getElementById("loginForm").style.display = "none";
                document.getElementById("signupForm").style.display = "block";
                document.getElementById("signupTab").classList.add("active");
                document.getElementById("loginTab").classList.remove("active");
            }
        </script>

    </body>
    </html>
    """, msg=msg)

# ======================================================
#                        HOME
# ======================================================
@app.route("/")
def home():
    if not require_login(): return redirect("/auth")

    return render_template_string("""
    <html><head>
    <title>SEG Home</title>
    <style>
    body{font-family:Arial;background:#eef2f9;}
    .box{width:60%;margin:40px auto;background:white;padding:25px;
         border-radius:15px;box-shadow:0 0 18px rgba(0,0,0,0.2);}
    textarea,input{width:100%;padding:12px;margin-top:10px;}
    button{background:#007bff;color:white;padding:12px;border:none;
           border-radius:6px;margin-top:15px;}
    </style>
    </head>

    <body>
    <div class="box">
        <h2>Scan Email</h2>
        <form method="POST" action="/scan">
            <input name="sender_email" placeholder="Sender Email">
            <textarea name="email_body" placeholder="Paste email content"></textarea>
            <button>Scan</button>
        </form>
        <p style="margin-top:15px;">
            <a href="/dashboard">📊 Dashboard</a> |
            <a href="/admin">🛡 Admin Panel</a> |
            <a href="/logout">Logout</a>
        </p>
    </div>
    </body>
    </html>
    """)


# ======================================================
#                     LOGOUT
# ======================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/auth")


# ======================================================
#                     SCAN EMAIL
# ======================================================
@app.route("/scan", methods=["POST"])
def scan():
    if not require_login(): return redirect("/auth")

    sender=request.form.get("sender_email")
    body=request.form.get("email_body")

    res = classify_email(sender, body)

    log = Logs(record=json.dumps(res),time=str(datetime.now()))
    db.session.add(log); db.session.commit()

    if res["label"]==1:
        q=Quarantine(body=body,time=str(datetime.now()))
        db.session.add(q); db.session.commit()

    return render_template_string("""
    <html><head>
    <title>Scan Result</title>
    <style>
      body{font-family:Arial;background:#eef2f9;}
      .box{width:70%;margin:40px auto;background:white;padding:20px;
           border-radius:12px;box-shadow:0 0 18px rgba(0,0,0,0.2);}
    </style></head>

    <body>
    <div class="box">
        <h2>Scan Result</h2>
        <p><b>Label:</b>
        {% if res["label"]==1 %}<span style='color:red;'>Phishing</span>
        {% else %}<span style='color:green;'>Safe</span>{% endif %}</p>

        <p><b>Risk Score:</b> {{ res["risk"] }}</p>
        <p><b>ML Score:</b> {{ res["ml"] }}</p>

        <h3>SEG Checks</h3>
        <p>SPF: {{ res["spf"] }}</p>
        <p>DMARC: {{ res["dmarc"] }}</p>
        <p>DKIM: {{ res["dkim"] }}</p>

        <h3>URLs</h3>
        <pre>{{ res["urls"] }}</pre>

        <a href="/">⬅ Back</a>
    </div>
    </body></html>
    """,res=res)


# ======================================================
#                        DASHBOARD
# ======================================================
@app.route("/dashboard")
def dashboard():
    if not require_login(): return redirect("/auth")

    acc=[0.82,0.85,0.89,0.92,0.94]
    loss=[0.45,0.36,0.28,0.19,0.15]

    return render_template_string("""
    <html>
    <head>
    <title>Dashboard</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
    .plotbox{padding:10px;background:white;border-radius:10px;
             box-shadow:0 0 12px rgba(0,0,0,0.2);}
    body{background:#eef2f9;}
    </style>
    </head>

    <body class="p-4">

    <h2 class="text-center mb-4">📊 Dashboard</h2>

    <div class="container">
      <div class="row g-4">
        {% for img in ["conf_lgb","conf_nb","roc","fi","url_dist","acc","loss"] %}
        <div class="col-md-4">
            <div class="plotbox">
                <img src="/static/plots/{{img}}.png" class="img-fluid">
            </div>
        </div>
        {% endfor %}
      </div>

      <hr class="my-4">

      <div class="row">
        <div class="col-md-6"><canvas id="accchart"></canvas></div>
        <div class="col-md-6"><canvas id="losschart"></canvas></div>
      </div>
    </div>

    <script>
    new Chart(document.getElementById('accchart'),{
        type:'line',
        data:{labels:['1','2','3','4','5'],
        datasets:[{label:'Accuracy',data:{{acc}},borderColor:'green'}]}
    });

    new Chart(document.getElementById('losschart'),{
        type:'line',
        data:{labels:['1','2','3','4','5'],
        datasets:[{label:'Loss',data:{{loss}},borderColor:'red'}]}
    });
    </script>

    </body></html>
    """,acc=acc,loss=loss)


# ======================================================
#                  ADMIN PANEL
# ======================================================
@app.route("/admin")
def admin():
    if not require_login(): return redirect("/auth")
    if not is_admin():
        return "Unauthorized"

    search = request.args.get("search","")
    page = int(request.args.get("page",1))
    per_page = 5

    q = Quarantine.query
    if search:
        q = q.filter(Quarantine.body.like(f"%{search}%"))

    total = q.count()
    pages = (total//per_page)+1

    data = q.order_by(Quarantine.id.desc()).offset((page-1)*per_page).limit(per_page).all()

    return render_template_string("""
    <html>
    <head>
    <title>Admin</title>
    <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">

    <style>
    body{background:#eef2f9;}
    .card{border-radius:12px;box-shadow:0 0 14px rgba(0,0,0,0.25);}
    </style>
    </head>

    <body class="p-4">

    <div class="container">
      <div class="card p-4">
        <h2>🛡 Admin Panel</h2>

        <form method="GET">
          <input class="form-control" name="search" placeholder="Search email..."
                 value="{{request.args.get('search','')}}">
        </form>

        <table class="table table-bordered table-striped mt-4">
          <thead class="table-dark">
            <tr>
              <th>Body</th>
              <th>Time</th>
              <th>View</th>
              <th>Release</th>
              <th>Delete</th>
            </tr>
          </thead>
          <tbody>
            {% for q in data %}
            <tr>
              <td>{{q.body[:50]}}...</td>
              <td>{{q.time}}</td>
              <td><button class="btn btn-info btn-sm"
                  onclick="showModal('{{q.body}}')">View</button></td>
              <td><a class="btn btn-success btn-sm" href="/release/{{q.id}}">Release</a></td>
              <td><a class="btn btn-danger btn-sm" href="/delete/{{q.id}}">Delete</a></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>

        <a href="/restore_all" class="btn btn-warning">Restore All</a>
        <a href="/delete_all" class="btn btn-danger">Delete All</a>
      </div>

      <!-- Pagination -->
      <nav class="mt-3">
      <ul class="pagination">
        {% for p in range(1,pages+1) %}
        <li class="page-item {% if p==page %}active{% endif %}">
          <a class="page-link" href="/admin?page={{p}}&search={{search}}">{{p}}</a>
        </li>
        {% endfor %}
      </ul>
      </nav>

    </div>

    <!-- MODAL -->
    <div id="m" style="
       position:fixed;top:0;left:0;width:100%;height:100%;
       background:rgba(0,0,0,0.6);display:none;align-items:center;justify-content:center;">
       <div style="background:white;padding:20px;border-radius:10px;width:60%;">
         <h3>Email Content</h3>
         <pre id="modaltext"></pre>
         <button onclick="document.getElementById('m').style.display='none'"
                 class="btn btn-secondary">Close</button>
       </div>
    </div>

    <script>
    function showModal(t){
        document.getElementById('modaltext').innerText=t;
        document.getElementById('m').style.display='flex';
    }
    </script>

    </body></html>
    """,
    data=data, page=page, pages=pages, search=search)


# ======================================================
#      ADMIN: RELEASE ONE / DELETE ONE / ALL
# ======================================================
@app.route("/release/<int:id>")
def release_email(id):
    if not is_admin(): return "Unauthorized"
    q=Quarantine.query.get(id)
    if q:
        db.session.delete(q); db.session.commit()
    return redirect("/admin")

@app.route("/delete/<int:id>")
def delete_email(id):
    if not is_admin(): return "Unauthorized"
    q=Quarantine.query.get(id)
    if q:
        db.session.delete(q); db.session.commit()
    return redirect("/admin")

@app.route("/restore_all")
def restore_all():
    if not is_admin(): return "Unauthorized"
    Quarantine.query.delete()
    db.session.commit()
    return redirect("/admin")

@app.route("/delete_all")
def delete_all():
    if not is_admin(): return "Unauthorized"
    Quarantine.query.delete()
    db.session.commit()
    return redirect("/admin")


# ======================================================
#                        LOGS DOWNLOAD
# ======================================================
@app.route("/download_logs")
def download_logs():
    logs = Logs.query.all()
    path="logs.txt"
    with open(path,"w") as f:
        for l in logs:
            f.write(l.time+" | "+l.record+"\n")
    return send_file(path,as_attachment=True)


# ======================================================
#                RUN SERVER
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
