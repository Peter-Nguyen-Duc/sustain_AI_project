

"""
Nutrition Agent — Flask starter application

Install:
    pip install Flask

Run:
    export FLASK_SECRET_KEY='replace-with-a-long-random-secret'
    python nutrition_agent_app.py

Replace rag_generate_response() with the existing agent/RAG integration. The function
receives the user's prompt, compact profile context, and recent conversation messages.
Never put model/API keys in this file or in client-side JavaScript.
"""

import json
import os
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, abort, flash, g, jsonify, redirect, render_template_string, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from blinders_framework import diet_planner_agentic_agent


try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # Jupyter, Colab, and VS Code notebook execution
    BASE_DIR = os.getcwd()
DATABASE = os.path.join(BASE_DIR, "nutrition_agent.db")
APP_SECRET = os.environ.get("FLASK_SECRET_KEY")

if not APP_SECRET:
    # Convenient for a local demo; use a persistent environment secret in deployment.
    APP_SECRET = secrets.token_urlsafe(48)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=APP_SECRET,
    DATABASE=DATABASE,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "false").lower() == "true",
    MAX_CONTENT_LENGTH=64 * 1024,
)



# -----------------------------------------------------------------------------
# Database and security helpers
# -----------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS profiles (
        user_id INTEGER PRIMARY KEY,
        age_group TEXT NOT NULL DEFAULT '',
        sex TEXT NOT NULL DEFAULT '',
        height_cm REAL,
        weight_kg REAL,
        activity_level TEXT NOT NULL DEFAULT '',
        goal TEXT NOT NULL DEFAULT '',
        dietary_pattern TEXT NOT NULL DEFAULT '',
        allergies TEXT NOT NULL DEFAULT '',
        disliked_foods TEXT NOT NULL DEFAULT '',
        medical_notes TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT 'New nutrition conversation',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    """
    with closing(sqlite3.connect(app.config["DATABASE"])) as db:
        db.executescript(schema)
        db.commit()


def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def validate_csrf(value):
    expected = session.get("csrf_token", "")
    return bool(expected and value and secrets.compare_digest(expected, value))


def require_csrf_api():
    if not validate_csrf(request.headers.get("X-CSRF-Token", "")):
        return jsonify({"error": "Your session token is invalid or expired. Refresh and try again."}), 403
    return None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please sign in to continue."}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    if "user_id" not in session:
        return None
    return get_db().execute(
        "SELECT id, name, email, created_at FROM users WHERE id = ?", (session["user_id"],)
    ).fetchone()


def profile_for(user_id):
    return get_db().execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()


def serialize_profile(profile):
    if not profile:
        return {}
    return {key: profile[key] for key in profile.keys() if key != "user_id"}


def clean_text(value, maximum=500):
    return str(value or "").strip()[:maximum]


def parse_optional_number(value, low, high, label):
    raw = str(value or "").strip()
    if not raw:
        return None, None
    try:
        number = float(raw)
    except ValueError:
        return None, f"{label} must be a number."
    if not low <= number <= high:
        return None, f"{label} must be between {low} and {high}."
    return number, None


def conversation_owned_by(user_id, conversation_id):
    return get_db().execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
    ).fetchone()


def create_conversation(user_id, title="New nutrition conversation"):
    now = utc_now()
    cursor = get_db().execute(
        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user_id, title, now, now),
    )
    get_db().commit()
    return cursor.lastrowid


def as_message(row):
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }


# -----------------------------------------------------------------------------
# Replace this adapter with your actual RAG/agent call.
# -----------------------------------------------------------------------------
def rag_generate_response(message, profile_context=None, conversation_history=None):
    """Development fallback. Replace the body with the existing agent pipeline.

    Recommended agent-side constraints:
    - Treat allergies and medical notes as safety-relevant context, not diagnoses.
    - State uncertainty and recommend a registered dietitian/clinician for clinical needs.
    - Do not prescribe treatment, diagnose, or encourage unsafe restriction.
    """


    MODEL_NAME = "qwen3.6"

    diet_planner_agentic_system = diet_planner_agentic_agent(MODEL_NAME)


    prompt = f"""
    ### call content:
    {message}

    ### profile context:
    {profile_context or "No profile context provided."}

    ### conversation history:
    {conversation_history or "No conversation history provided."}
    """

    # query = "I want a diet plan for a week, I want to lose weight and I want to eat healthy food"
    response_val = diet_planner_agentic_system.diet_chat(prompt)
    return response_val


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------
AUTH_TEMPLATE = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }} | Nourish AI</title><style>
:root{--ink:#17251d;--muted:#617168;--forest:#176447;--lime:#b8ed8d;--paper:#fbfcf7;--line:#dce5d9;--danger:#a72f3d}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;color:var(--ink);font:15px Inter,system-ui,sans-serif;background:radial-gradient(circle at 10% 10%,#dff7c9,transparent 29%),radial-gradient(circle at 92% 92%,#c9f0dc,transparent 30%),var(--paper)}.card{width:min(100%,470px);padding:36px;background:#fff;border:1px solid var(--line);border-radius:24px;box-shadow:0 22px 70px #173b2717}.brand{display:flex;align-items:center;gap:11px;font-weight:800;font-size:20px}.logo{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:var(--forest);color:#fff}.sub{color:var(--muted);line-height:1.6;margin:12px 0 26px}label{display:block;font-size:13px;font-weight:700;margin:15px 0 6px}input{width:100%;padding:12px;border:1px solid var(--line);border-radius:10px;font:inherit;outline:0}input:focus{border-color:var(--forest);box-shadow:0 0 0 3px #17644718}button{width:100%;margin-top:22px;padding:13px;border:0;border-radius:11px;background:var(--forest);color:#fff;font:700 15px inherit;cursor:pointer}.notice{margin:15px 0;padding:11px;border-radius:9px;background:#fff2f3;color:var(--danger)}.switch{text-align:center;color:var(--muted);margin:20px 0 0}.switch a{color:var(--forest);font-weight:700;text-decoration:none}.privacy{margin-top:24px;color:var(--muted);font-size:12px;line-height:1.5}
</style></head><body><main class="card"><div class="brand"><span class="logo">✦</span>Nourish AI</div><p class="sub">{{ subtitle }}</p>{% with messages=get_flashed_messages() %}{% for message in messages %}<div class="notice">{{ message }}</div>{% endfor %}{% endwith %}{{ body|safe }}</main></body></html>
"""


@app.route("/")
def index():
    return redirect(url_for("app_home" if "user_id" in session else "login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("app_home"))
    if request.method == "POST":
        if not validate_csrf(request.form.get("csrf_token")):
            abort(400, "Invalid form token.")
        name = clean_text(request.form.get("name"), 80)
        email = clean_text(request.form.get("email"), 254).lower()
        password = request.form.get("password", "")
        if len(name) < 2:
            flash("Please enter your name.")
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Please enter a valid email address.")
        elif len(password) < 12:
            flash("Use a password of at least 12 characters.")
        else:
            try:
                now = utc_now()
                cur = get_db().execute(
                    "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (name, email, generate_password_hash(password), now),
                )
                get_db().execute("INSERT INTO profiles (user_id, updated_at) VALUES (?, ?)", (cur.lastrowid, now))
                get_db().commit()
                session.clear()
                session["user_id"] = cur.lastrowid
                csrf_token()
                return redirect(url_for("app_home", onboarding="1"))
            except sqlite3.IntegrityError:
                flash("An account with that email already exists. Please sign in.")
    body = f'''<form method="post" novalidate><input type="hidden" name="csrf_token" value="{csrf_token()}"><label for="name">Your name</label><input id="name" name="name" required maxlength="80" autocomplete="name"><label for="email">Email</label><input id="email" type="email" name="email" required autocomplete="email"><label for="password">Password</label><input id="password" type="password" name="password" required minlength="12" autocomplete="new-password"><button>Create account</button></form><p class="switch">Already have an account? <a href="{url_for('login')}">Sign in</a></p><p class="privacy">Your profile is used to tailor non-clinical nutrition guidance. Do not enter medical records or emergency information.</p>'''
    return render_template_string(AUTH_TEMPLATE, title="Create account", subtitle="Create your private nutrition profile and start a conversation.", body=body)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("app_home"))
    if request.method == "POST":
        if not validate_csrf(request.form.get("csrf_token")):
            abort(400, "Invalid form token.")
        email = clean_text(request.form.get("email"), 254).lower()
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session["user_id"] = user["id"]
            csrf_token()
            return redirect(url_for("app_home"))
        flash("Email or password is incorrect.")
    body = f'''<form method="post"><input type="hidden" name="csrf_token" value="{csrf_token()}"><label for="email">Email</label><input id="email" type="email" name="email" required autocomplete="email"><label for="password">Password</label><input id="password" type="password" name="password" required autocomplete="current-password"><button>Sign in</button></form><p class="switch">New to Nourish AI? <a href="{url_for('register')}">Create an account</a></p>'''
    return render_template_string(AUTH_TEMPLATE, title="Sign in", subtitle="Welcome back. Continue your nutrition conversations securely.", body=body)


@app.post("/logout")
@login_required
def logout():
    if not validate_csrf(request.form.get("csrf_token")):
        abort(400, "Invalid form token.")
    session.clear()
    return redirect(url_for("login"))


APP_TEMPLATE = r"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nourish AI</title>
<style>
:root{--bg:#f6f8f3;--panel:#fff;--ink:#17251d;--muted:#617168;--line:#dce5d9;--forest:#176447;--forest2:#0e4b35;--mint:#e6f7e7;--lime:#b8ed8d;--warn:#fff7de;--danger:#a72f3d;--shadow:0 12px 34px #173b2712}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px Inter,ui-sans-serif,system-ui,sans-serif}.shell{display:grid;grid-template-columns:270px minmax(0,1fr);height:100vh}.side{display:flex;flex-direction:column;padding:20px 14px;background:#f0f5ee;border-right:1px solid var(--line)}.brand{display:flex;align-items:center;gap:10px;padding:5px 9px 24px;font-weight:800;font-size:19px}.logo{display:grid;place-items:center;width:35px;height:35px;border-radius:11px;background:var(--forest);color:#fff}.new{width:100%;padding:12px;border:0;border-radius:11px;background:var(--forest);color:#fff;font:700 14px inherit;cursor:pointer}.label{margin:26px 10px 8px;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.history{overflow:auto}.thread{width:100%;padding:10px;text-align:left;color:var(--ink);background:transparent;border:0;border-radius:9px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.thread:hover,.thread.active{background:#fff}.profile-link{margin-top:auto;width:100%;padding:11px;text-align:left;background:#fff;border:1px solid var(--line);border-radius:11px;color:var(--ink);cursor:pointer}.profile-link small{display:block;color:var(--muted);margin-top:3px}.main{display:flex;min-width:0;flex-direction:column}.top{display:flex;align-items:center;justify-content:space-between;min-height:70px;padding:0 27px;background:#ffffffc7;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}.title{font-weight:750;font-size:15px}.tag{margin-left:8px;padding:4px 7px;color:var(--forest);background:var(--mint);border-radius:999px;font-size:10px;font-weight:800;letter-spacing:.07em;text-transform:uppercase}.user-menu{display:flex;align-items:center;gap:9px}.avatar{display:grid;place-items:center;width:32px;height:32px;border-radius:50%;background:var(--lime);color:var(--forest2);font-weight:800}.logout{padding:8px;background:transparent;border:1px solid var(--line);border-radius:8px;cursor:pointer}.scroll{flex:1;overflow:auto}.conversation{width:min(900px,calc(100% - 34px));min-height:100%;margin:auto;padding:32px 0}.welcome{max-width:700px;margin:45px auto;text-align:center}.welcome h1{font-size:32px;letter-spacing:-.04em;margin:14px 0 10px}.welcome p{color:var(--muted);line-height:1.65}.leaf{display:grid;place-items:center;width:62px;height:62px;margin:auto;border-radius:20px;background:var(--lime);font-size:28px}.suggestions{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:26px}.suggestion{padding:14px;text-align:left;background:#fff;border:1px solid var(--line);border-radius:13px;color:var(--ink);cursor:pointer}.suggestion:hover{border-color:#80b994;background:#fbfffa}.suggestion span{display:block;color:var(--forest);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}.row{display:flex;gap:10px;margin:18px 0}.row.user{flex-direction:row-reverse}.bubble-wrap{max-width:min(78%,700px)}.meta{margin:0 2px 5px;color:var(--muted);font-size:11px}.row.user .meta{text-align:right}.bubble{padding:12px 14px;background:#fff;border:1px solid var(--line);border-radius:5px 15px 15px 15px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}.row.user .bubble{color:#fff;background:var(--forest);border-radius:15px 5px 15px 15px}.notice{margin:4px auto 20px;padding:11px 13px;max-width:900px;background:var(--warn);border:1px solid #f1dc9d;border-radius:10px;color:#635321;font-size:12px;line-height:1.5}.composer-area{padding:11px 20px 18px;background:linear-gradient(transparent,var(--bg) 40%)}.composer{display:flex;gap:9px;width:min(900px,100%);margin:auto;padding:8px 9px 8px 14px;background:#fff;border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}textarea{flex:1;max-height:150px;min-height:29px;padding:6px 0;resize:none;border:0;outline:0;font:inherit;line-height:1.5;background:transparent}.send{width:40px;height:40px;border:0;border-radius:10px;background:var(--forest);color:#fff;font-size:17px;cursor:pointer}.send:disabled{opacity:.5}.hint{margin:8px;text-align:center;color:var(--muted);font-size:11px}.modal{position:fixed;inset:0;z-index:4;display:none;place-items:center;padding:18px;background:#173b2740}.modal.show{display:grid}.card{width:min(100%,650px);max-height:90vh;overflow:auto;padding:25px;background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}.card-head{display:flex;align-items:start;justify-content:space-between;gap:12px}.card h2{margin:0;font-size:20px}.card p{color:var(--muted);line-height:1.55}.close{border:0;background:transparent;font-size:24px;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:13px}.full{grid-column:1/-1}label{display:block;font-size:12px;font-weight:750;margin-bottom:5px}input,select{width:100%;padding:10px;border:1px solid var(--line);border-radius:9px;background:#fff;font:inherit}input:focus,select:focus,textarea:focus{outline:2px solid #8bc7a211;outline-offset:1px;border-color:#80b994}.save{margin-top:19px;padding:11px 15px;border:0;border-radius:9px;background:var(--forest);color:white;font-weight:750;cursor:pointer}.toast{position:fixed;right:20px;bottom:20px;padding:11px 14px;background:#1b2f24;color:#fff;border-radius:10px;opacity:0;transform:translateY(8px);transition:.2s;pointer-events:none}.toast.show{opacity:1;transform:none}@media(max-width:760px){.shell{display:block;height:100dvh}.side{display:none}.top{padding:0 14px}.conversation{width:min(100% - 24px,900px);padding-top:18px}.suggestions{grid-template-columns:1fr}.bubble-wrap{max-width:85%}.grid{grid-template-columns:1fr}.full{grid-column:auto}.logout{font-size:0}.logout::after{content:'↪';font-size:16px}.welcome h1{font-size:27px}}
</style></head><body>
<div class="shell"><aside class="side"><div class="brand"><span class="logo">✦</span>Nourish AI</div><button class="new" id="newChat">＋ New conversation</button><div class="label">Your conversations</div><div class="history" id="history"></div><button class="profile-link" id="openProfile"><strong>{{ user['name'] }}</strong><small>Nutrition profile & preferences</small></button></aside>
<main class="main"><header class="top"><div><span class="title" id="title">Nutrition conversation</span><span class="tag">Personalised</span></div><div class="user-menu"><span class="avatar">{{ user['name'][0]|upper }}</span><form method="post" action="{{ url_for('logout') }}"><input type="hidden" name="csrf_token" value="{{ csrf }}"><button class="logout" title="Sign out">Sign out</button></form></div></header>
<section class="scroll" id="scroll"><div class="conversation" id="conversation"><section class="welcome" id="welcome"><div class="leaf">🥬</div><h1>Eat well, your way.</h1><p>Ask for practical meal ideas, grocery lists, recipes, and habit support based on the preferences in your profile.</p><div class="suggestions"><button class="suggestion" data-prompt="Create a balanced 3-day meal plan using my dietary preferences."><span>Meal plan</span>Create a balanced 3-day meal plan.</button><button class="suggestion" data-prompt="Suggest five quick high-protein breakfast ideas that respect my profile."><span>Ideas</span>Suggest quick high-protein breakfasts.</button><button class="suggestion" data-prompt="Make a budget-friendly grocery list for balanced weekday dinners."><span>Groceries</span>Make a budget-friendly grocery list.</button><button class="suggestion" data-prompt="Help me build one small nutrition habit for this week."><span>Habit</span>Build a small nutrition habit.</button></div></section></div></section>
<div class="notice">Nourish AI provides general educational nutrition guidance, not medical advice. It is not a substitute for a registered dietitian or clinician.</div><footer class="composer-area"><form class="composer" id="chatForm"><textarea id="input" rows="1" maxlength="4000" placeholder="Ask about meals, recipes, shopping, or nutrition habits…" aria-label="Nutrition question"></textarea><button class="send" id="send" aria-label="Send">↑</button></form><div class="hint">Enter to send · Shift + Enter for a new line</div></footer></main></div>
<div class="modal" id="profileModal" aria-hidden="true"><section class="card" role="dialog" aria-modal="true" aria-labelledby="profileHeading"><div class="card-head"><div><h2 id="profileHeading">Your nutrition profile</h2><p>Use only the details you want the agent to consider. Avoid confidential medical information.</p></div><button class="close" id="closeProfile" aria-label="Close">×</button></div><form id="profileForm"><div class="grid"><div><label for="age_group">Age group</label><select id="age_group" name="age_group"><option value="">Prefer not to say</option><option>18–24</option><option>25–34</option><option>35–44</option><option>45–54</option><option>55–64</option><option>65+</option></select></div><div><label for="sex">Sex (optional)</label><select id="sex" name="sex"><option value="">Prefer not to say</option><option>Female</option><option>Male</option><option>Intersex</option><option>Prefer to self-describe</option></select></div><div><label for="height_cm">Height (cm, optional)</label><input id="height_cm" name="height_cm" type="number" min="80" max="250" step="0.1"></div><div><label for="weight_kg">Weight (kg, optional)</label><input id="weight_kg" name="weight_kg" type="number" min="25" max="350" step="0.1"></div><div><label for="activity_level">Activity level</label><select id="activity_level" name="activity_level"><option value="">Not specified</option><option>Mostly sedentary</option><option>Lightly active</option><option>Moderately active</option><option>Very active</option></select></div><div><label for="goal">Current goal</label><select id="goal" name="goal"><option value="">Not specified</option><option>Balanced eating</option><option>Weight management</option><option>Build muscle</option><option>Improve energy</option><option>Meal planning</option></select></div><div class="full"><label for="dietary_pattern">Dietary pattern</label><input id="dietary_pattern" name="dietary_pattern" maxlength="160" placeholder="e.g., vegetarian, halal, Mediterranean, no preference"></div><div class="full"><label for="allergies">Allergies or foods to avoid</label><input id="allergies" name="allergies" maxlength="300" placeholder="e.g., peanuts, shellfish, lactose"></div><div class="full"><label for="disliked_foods">Disliked foods / practical constraints</label><input id="disliked_foods" name="disliked_foods" maxlength="300" placeholder="e.g., no fish, 20-minute dinners, limited budget"></div><div class="full"><label for="medical_notes">Safety note (optional)</label><input id="medical_notes" name="medical_notes" maxlength="300" placeholder="Only brief constraints you want considered; do not add clinical records"></div></div><button class="save" type="submit">Save profile</button></form></section></div><div class="toast" id="toast"></div>
<script>
const csrf={{ csrf|tojson }}; const state={conversationId:null,waiting:false};
const $=id=>document.getElementById(id); const input=$("input"), send=$("send"), conversation=$("conversation"), scroll=$("scroll"), welcome=$("welcome"), title=$("title"), toast=$("toast");
function say(text){toast.textContent=text;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),2600)}
function api(path,options={}){options.headers={...(options.headers||{}),'Content-Type':'application/json','X-CSRF-Token':csrf};return fetch(path,options)}
function time(iso){return new Date(iso||Date.now()).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}
function scrollBottom(){requestAnimationFrame(()=>scroll.scrollTop=scroll.scrollHeight)}
function resize(){input.style.height='auto';input.style.height=Math.min(input.scrollHeight,150)+'px'}
function addMessage(role,content,createdAt){const row=document.createElement('article');row.className='row '+(role==='assistant'?'assistant':'user');const icon=document.createElement('div');icon.className='avatar';icon.textContent=role==='assistant'?'✦':'{{ user['name'][0]|upper }}';const wrap=document.createElement('div');wrap.className='bubble-wrap';const meta=document.createElement('div');meta.className='meta';meta.textContent=(role==='assistant'?'Nourish AI':'You')+' · '+time(createdAt);const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=content;wrap.append(meta,bubble);row.append(icon,wrap);conversation.append(row);scrollBottom();return row}
function typing(){const row=addMessage('assistant','Thinking…',new Date().toISOString());row.id='typing';return row}
async function refreshHistory(){const r=await api('/api/conversations');if(!r.ok)return;const data=await r.json();const host=$('history');host.innerHTML='';data.conversations.forEach(c=>{const b=document.createElement('button');b.className='thread'+(c.id===state.conversationId?' active':'');b.textContent=c.title;b.onclick=()=>loadConversation(c.id);host.append(b)})}
async function createConversation(){const r=await api('/api/conversations',{method:'POST',body:JSON.stringify({})});const d=await r.json();if(!r.ok)throw new Error(d.error);state.conversationId=d.conversation.id;return d.conversation}
async function loadConversation(id){const r=await api('/api/conversations/'+id);const d=await r.json();if(!r.ok){say(d.error||'Could not load conversation');return}state.conversationId=id;title.textContent=d.conversation.title;document.querySelectorAll('.row').forEach(e=>e.remove());welcome.style.display=d.messages.length?'none':'block';d.messages.forEach(m=>addMessage(m.role,m.content,m.created_at));refreshHistory()}
async function sendMessage(text){const message=text.trim();if(!message||state.waiting)return;state.waiting=true;send.disabled=true;welcome.style.display='none';try{if(!state.conversationId)await createConversation();addMessage('user',message,new Date().toISOString());input.value='';resize();const pending=typing();const r=await api('/api/chat',{method:'POST',body:JSON.stringify({conversation_id:state.conversationId,message})});const d=await r.json();pending.remove();if(!r.ok)throw new Error(d.error||'The agent could not respond.');addMessage('assistant',d.reply,d.created_at);title.textContent=d.conversation_title;refreshHistory()}catch(err){const pending=$('typing');if(pending)pending.remove();addMessage('assistant','I could not process that request. Please try again in a moment.');say(err.message)}finally{state.waiting=false;send.disabled=false;input.focus()}}
$('chatForm').addEventListener('submit',e=>{e.preventDefault();sendMessage(input.value)});input.addEventListener('input',resize);input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('chatForm').requestSubmit()}});document.querySelectorAll('.suggestion').forEach(b=>b.onclick=()=>{input.value=b.dataset.prompt;resize();input.focus()});$('newChat').onclick=async()=>{if(state.waiting)return say('Please wait for the response.');state.conversationId=null;document.querySelectorAll('.row').forEach(e=>e.remove());welcome.style.display='block';title.textContent='Nutrition conversation';input.focus();refreshHistory()};
$('openProfile').onclick=async()=>{const r=await api('/api/profile');const d=await r.json();Object.entries(d.profile||{}).forEach(([k,v])=>{const el=document.querySelector('[name="'+k+'"]');if(el&&v!==null)el.value=v});$('profileModal').classList.add('show');$('profileModal').setAttribute('aria-hidden','false')};function closeProfile(){$('profileModal').classList.remove('show');$('profileModal').setAttribute('aria-hidden','true')} $('closeProfile').onclick=closeProfile;$('profileModal').onclick=e=>{if(e.target===$('profileModal'))closeProfile()};$('profileForm').addEventListener('submit',async e=>{e.preventDefault();const values=Object.fromEntries(new FormData(e.target));const r=await api('/api/profile',{method:'PUT',body:JSON.stringify(values)});const d=await r.json();if(!r.ok)return say(d.error||'Could not save profile.');closeProfile();say('Profile saved. New replies will use it as context.')});refreshHistory();input.focus();
</script></body></html>
"""


@app.route("/app")
@login_required
def app_home():
    user = current_user()
    return render_template_string(APP_TEMPLATE, user=user, csrf=csrf_token())


# -----------------------------------------------------------------------------
# JSON API — all records are filtered by authenticated owner.
# -----------------------------------------------------------------------------
@app.get("/api/profile")
@login_required
def get_profile():
    return jsonify({"profile": serialize_profile(profile_for(session["user_id"]))})


@app.put("/api/profile")
@login_required
def update_profile():
    blocked = require_csrf_api()
    if blocked:
        return blocked
    data = request.get_json(silent=True) or {}
    height, error = parse_optional_number(data.get("height_cm"), 80, 250, "Height")
    if error:
        return jsonify({"error": error}), 400
    weight, error = parse_optional_number(data.get("weight_kg"), 25, 350, "Weight")
    if error:
        return jsonify({"error": error}), 400
    values = (
        clean_text(data.get("age_group"), 30), clean_text(data.get("sex"), 40), height, weight,
        clean_text(data.get("activity_level"), 40), clean_text(data.get("goal"), 60),
        clean_text(data.get("dietary_pattern"), 160), clean_text(data.get("allergies"), 300),
        clean_text(data.get("disliked_foods"), 300), clean_text(data.get("medical_notes"), 300),
        utc_now(), session["user_id"],
    )
    get_db().execute("""UPDATE profiles SET age_group=?, sex=?, height_cm=?, weight_kg=?, activity_level=?,
        goal=?, dietary_pattern=?, allergies=?, disliked_foods=?, medical_notes=?, updated_at=? WHERE user_id=?""", values)
    get_db().commit()
    return jsonify({"profile": serialize_profile(profile_for(session["user_id"]))})


@app.get("/api/conversations")
@login_required
def list_conversations():
    rows = get_db().execute("""SELECT id, title, created_at, updated_at FROM conversations
        WHERE user_id=? ORDER BY updated_at DESC LIMIT 50""", (session["user_id"],)).fetchall()
    return jsonify({"conversations": [dict(row) for row in rows]})


@app.post("/api/conversations")
@login_required
def new_conversation():
    blocked = require_csrf_api()
    if blocked:
        return blocked
    conv_id = create_conversation(session["user_id"])
    row = conversation_owned_by(session["user_id"], conv_id)
    return jsonify({"conversation": dict(row)}), 201


@app.get("/api/conversations/<int:conversation_id>")
@login_required
def get_conversation(conversation_id):
    conversation = conversation_owned_by(session["user_id"], conversation_id)
    if not conversation:
        return jsonify({"error": "Conversation not found."}), 404
    rows = get_db().execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
    return jsonify({"conversation": dict(conversation), "messages": [as_message(row) for row in rows]})


@app.post("/api/chat")
@login_required
def chat():
    blocked = require_csrf_api()
    if blocked:
        return blocked
    data = request.get_json(silent=True) or {}
    message = clean_text(data.get("message"), 4000)
    if not message:
        return jsonify({"error": "Please enter a message before sending."}), 400
    try:
        conversation_id = int(data.get("conversation_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "A valid conversation is required."}), 400
    conversation = conversation_owned_by(session["user_id"], conversation_id)
    if not conversation:
        return jsonify({"error": "Conversation not found."}), 404

    db = get_db()
    now = utc_now()
    db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)", (conversation_id, message, now))
    recent = db.execute("""SELECT role, content FROM messages WHERE conversation_id=?
        ORDER BY id DESC LIMIT 12""", (conversation_id,)).fetchall()
    recent_history = [dict(row) for row in reversed(recent)]
    user = current_user()
    context = serialize_profile(profile_for(user["id"]))
    context["name"] = user["name"]

    try:
        reply = str(rag_generate_response(message, profile_context=context, conversation_history=recent_history)).strip()
    except Exception:
        app.logger.exception("Nutrition agent error")
        db.rollback()
        return jsonify({"error": "The nutrition agent is temporarily unavailable. Please try again."}), 502
    if not reply:
        reply = "I could not generate a helpful response this time. Please try rephrasing your question."

    db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)", (conversation_id, reply[:8000], utc_now()))
    title = conversation["title"]
    if title == "New nutrition conversation":
        title = message[:52] + ("…" if len(message) > 52 else "")
        db.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, utc_now(), conversation_id))
    else:
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (utc_now(), conversation_id))
    db.commit()
    return jsonify({"reply": reply[:8000], "created_at": utc_now(), "conversation_title": title})


if __name__ == "__main__":
    init_db()
    # debug=False prevents exposing the interactive debugger in a deployed service.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)

