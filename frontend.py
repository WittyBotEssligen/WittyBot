import streamlit as st
from WittyBot_V05_optimiert import WittyBotCore, AppConfig
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
import time
import streamlit.components.v1 as components
import base64

# Erstellt eine SQLite-Datenbank für allgemeines Feedback, falls sie noch nicht existiert
def init_feedback_db():
    db_path = Path("feedback.db")
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            stars INTEGER,
            comment TEXT,
            question TEXT,
            answer TEXT
        )
        """)
        conn.commit()
        conn.close()
init_feedback_db()

# 🧱 Initialisierung der Datenbank für Einzel-Feedbacks
def init_message_feedback_db():
    # Definiert den Pfad zur Datenbankdatei
    db_path = Path("message_feedback.db")

    # Falls die Datei noch nicht existiert, wird sie erstellt
    if not db_path.exists():
        # Verbindung zur neuen SQLite-Datenbank aufbauen
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Tabelle 'message_feedback' erstellen, wenn sie noch nicht existiert
        # Sie speichert:
        # - ID (automatisch hochgezählt)
        # - Zeitstempel des Feedbacks
        # - Die ursprüngliche Frage
        # - Die gegebene Antwort
        # - Die Bewertung (z. B. +1 oder -1)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS message_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            rating INTEGER
        )
        """)
        # Änderungen speichern und Verbindung schließen
        conn.commit()
        conn.close()
# Führt die Initialisierung direkt beim Start des Programms aus
init_message_feedback_db()

# 📋 Aktiviert das Logging-System für die Anwendung.
# Damit werden Laufzeitinformationen (wie Infos, Warnungen, Fehler) in der Konsole ausgegeben.
# Das ist sehr nützlich für die spätere Fehlersuche oder zur Überwachung der Bot-Aktivität.
logging.basicConfig(
    level=logging.INFO,  # Mindestlevel: Nur Meldungen ab "INFO" werden angezeigt (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format der Ausgabe: Zeitstempel – Level – Nachricht
    handlers=[logging.StreamHandler()]  # Ausgabeziel: Konsole (Standardausgabe)
)

# 🧭 Seitenkonfiguration festlegen
st.set_page_config(
    page_title="WittyBot – Dein Studienassistent",  # Titel im Browser-Tab
    page_icon="img/logo.png",                       # Icon im Browser-Tab (z. B. neben dem Titel)
    layout="wide"                                   # Breites Layout (mehr Platz für Inhalte)
)
    
# 💜 Stil (Individuelle Gestaltung des Interfaces über CSЫ)
# Das CSS wird in einem Markdown-Block direkt in den Streamlit-HTML-Code eingefügt.
st.markdown("""
    <style>
    /* 🧑‍🎓 Nachricht des Users: Hintergrund & Stil */
    .stChatMessage.user div:nth-child(2) {
        background-color: #E6E6FA !important;   /* Hell-lila Hintergrund */
        border-radius: 15px;                    /* Abgerundete Ecken */
        padding: 10px 15px;                     /* Innenabstand */
    }

    /* 🎨 Seitenleiste: Farbgebung */
    [data-testid="stSidebar"] {
        background-color: #F5F3FF !important;   /* Hell-violett */
    }

    /* 🕒 Zeitstempel-Stil */
    .timestamp {
        font-size: 11px;        /* Kleine Schriftgröße */
        color: #888;            /* Graue Farbe */
        margin-top: 4px;        /* Abstand nach oben */
    }

    /* ✍️ Eingabefeld: normaler Zustand */
    textarea {
        background-color: #F5F3FF !important;  /* Hellvioletter Hintergrund */
        transition: background-color 0.3s ease;  /* Weicher Farbübergang */
        border-radius: 10px !important;       /* Runde Ecken */
        color: black !important;              /* Schwarzer Text */
    }

    /* ✍️ Eingabefeld: wenn aktiv (fokussiert) */
    textarea:focus {
        background-color: #E0D4FA !important;  /* Etwas dunkler beim Klicken */
        outline: none !important;              /* Kein Standard-Fokusrahmen */
    }

    /* ❌ Stil für "Antwort abbrechen"-Knopf */
    .abort-button button {
        font-size: 12px !important;
        padding: 3px 8px !important;
        border-radius: 10px !important;
        background-color: #ffecec !important;  /* Rosa Hintergrund */
        color: #800000 !important;             /* Dunkelrot für Text */
        border: 1px solid #e0a0a0 !important;   /* Roter Rahmen */
        margin-top: 5px;
        margin-bottom: 10px;
    }

    /* 🧩 Sekundärbuttons (z. B. Feedback-Sterne) – kleinere Schrift */
    button[kind="secondary"] span {
        font-size: 12px !important;
    }

    /* 🧱 Hauptcontainer (Inhalt in der Mitte & verbreitert) */
    .block-container {
        max-width: 860px;             /* Max. Breite */
        padding-left: 2rem;           /* Abstand links */
        padding-right: 2rem;          /* Abstand rechts */
        margin: auto;                 /* Zentriert */
    }

    /* ⌨️ Eingabefeld im Chat an Hauptbreite anpassen */
    div[data-testid="stChatInput"] {
        max-width: 740px !important;  /* Gleiche Breite wie Inhalte */
        margin-left: auto !important;
        margin-right: auto !important;
    }

    </style>
""", unsafe_allow_html=True)


# 📚 Seitenleiste (Sidebar): enthält Logo, Titel, Feedback-Formular und ggf. Admin-Optionen
with st.sidebar:
    st.image("img/logo.png", width=100)  # Logo anzeigen
    st.title("📚 WittyBot")  # Titel des Bots in der Seitenleiste
    st.markdown("Dein intelligenter Studienassistent 🤖")  # Kurze Beschreibung

    # ⭐ Abschnitt für allgemeines Feedback zum gesamten Chat
    st.markdown("### 🌟 Feedback zum Gespräch")

    # Falls noch kein Feedback gegeben wurde, wird es erlaubt
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    # Nur anzeigen, wenn noch kein Feedback abgegeben wurde
    if not st.session_state.feedback_given:
        # ⭐ Sternebewertung: Skala von 1 (schlecht) bis 5 (sehr hilfreich)
        stars = st.slider("Wie hilfreich war das gesamte Gespräch?", 1, 5, 3)
        st.caption("1 = schlecht, 5 = sehr hilfreich")

        comment = ""
        # Wenn Bewertung ≤ 2: Kommentarfeld anzeigen
        if stars <= 2:
            comment = st.text_area("Was war unklar oder schlecht?")

        # 📩 Button zum Absenden des Feedbacks
        if st.button("📩 Feedback absenden"):

            messages = st.session_state.get("messages", [])  # Alle Chat-Nachrichten
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Zeitpunkt speichern
            entries = []  # Liste aller Einträge für die Datenbank

            # ➕ Alle Frage-Antwort-Paare zusammenstellen
            i = 0
            while i < len(messages) - 1:
                q = messages[i]
                a = messages[i + 1]

                if q["role"] == "user":
                    # Antwort-Text prüfen
                    answer_text = ""
                    if a["role"] == "assistant":
                        answer_text = a["content"]
                    else:
                        answer_text = "⚠️ Keine Antwort erhalten."  # Wenn keine Antwort vorhanden

                    # Feedback-Eintrag speichern
                    entries.append((timestamp, stars, comment, q["content"], answer_text))
                    i += 2  # Weiter zum nächsten Paar
                else:
                    i += 1

            # 🗃️ Speichern des Feedbacks in die Datenbank
            if entries:
                try:
                    conn = sqlite3.connect("feedback.db")  # Verbindung zur DB
                    cursor = conn.cursor()
                    cursor.executemany(
                        "INSERT INTO feedback (timestamp, stars, comment, question, answer) VALUES (?, ?, ?, ?, ?)",
                        entries  # Mehrere Einträge gleichzeitig einfügen
                    )
                    conn.commit()
                    conn.close()
                    st.session_state.feedback_given = True  # Kein erneutes Feedback möglich
                    st.success("✅ Danke für dein Feedback zum Gespräch!")  # Erfolgsmeldung
                except Exception as e:
                    st.error(f"❌ Fehler beim Speichern: {e}")  # Fehler beim Speichern
            else:
                st.warning("⚠️ Keine Frage-Antwort-Paare gefunden.")  # Kein Inhalt zum Speichern
    else:
        # Falls bereits Feedback gegeben wurde
        st.success("✅ Feedback wurde bereits abgeschickt.")

# 🗑️ Chat löschen – damit kann der Benutzer den gesamten bisherigen Verlauf zurücksetzen
    if st.button("🗑️ Chatverlauf löschen"):
        st.session_state.messages = []             # Alle bisherigen Chatnachrichten löschen
        st.session_state.aborted = False           # Zurücksetzen, falls vorher eine Antwort abgebrochen wurde
        st.session_state.is_thinking = False       # Bot ist nicht mehr „am Denken“
        st.session_state.feedback_mode = False     # Feedback-Modus wird deaktiviert
        st.session_state.feedback_incorrect = False  # Rückmeldung über falsche Antwort zurücksetzen
        st.rerun()  # 🔄 Seite neu laden, um den Zustand sofort sichtbar zu machen
     
    # 🔐 Admin-Bereich – nur für berechtigte Personen mit Passwort sichtbar
    st.markdown("---")  # Trennlinie im Seitenlayout
    st.subheader("🔐 Admin-Zugang")  # Abschnittstitel

    # Prüfen, ob der Admin-Status bereits in der Sitzung gespeichert ist
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    # ✅ Wenn Admin bereits eingeloggt ist:
    if st.session_state.is_admin:
        st.success("✅ Admin-Zugang aktiv")  # Grüne Erfolgsmeldung
        st.markdown("– Neue Dateien → in den Ordner `/data/` legen")  # Hinweis für Dokumentenpflege

        # 🔄 Möglichkeit zum Neuladen der Dokumentenbasis (Vektor-Datenbank)
        if st.button("🔄 Dokumente neu laden"):
            config = AppConfig()                    # Konfiguration neu laden
            bot = WittyBotCore(config)              # Bot mit aktueller Konfiguration starten
            bot.reload_vector_db()                  # Dokumente erneut verarbeiten
            st.success("Vektor-Datenbank wurde aktualisiert!")  # Erfolgsmeldung

        # 🔒 Ausloggen-Funktion
        if st.button("🔒 Ausloggen"):
            st.session_state.is_admin = False       # Admin-Modus deaktivieren
            st.success("Du wurdest ausgeloggt.")    # Bestätigung

    # ❌ Wenn nicht eingeloggt:
    else:
        # Passwort-Eingabe für Admin-Zugang
        admin_pw = st.text_input("Admin-Passwort eingeben:", type="password")

        # Klick auf "Einloggen"-Button
        if st.button("🔓 Einloggen"):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Aktuelle Zeit für Log
            try:
                ip = st.query_params.get("ip", ["unknown"])[0]  # IP-Adresse (optional)
            except:
                ip = "unknown"

            # 🔑 Prüfung des Passworts (hier hartkodiert!)
            if admin_pw == "ichbindumm":
                st.session_state.is_admin = True     # Admin-Rechte aktivieren
                st.success("✅ Admin-Zugang aktiviert!")  # Erfolgsmeldung

                # Erfolgreiches Login in Datei protokollieren
                with open("admin_logins.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{now}] Erfolgreich – IP: {ip}\n")
            else:
                st.error("❌ Falsches Passwort. Bitte wende dich an das Projektteam.")  # Fehleranzeige

                # Fehlgeschlagenes Login protokollieren
                with open("admin_logins.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{now}] Fehlgeschlagen – IP: {ip} – Passwort: {admin_pw}\n")

import hashlib

def generate_message_key(question, answer, timestamp):
    """Generiert eindeutigen Hash-Schlüssel für eine Frage-Antwort-Kombination zu einem bestimmten Zeitpunkt.
    Dies dient dazu, Feedback eindeutig zu identifizieren und doppelte Bewertungen zu verhindern.
    Die Kombination aus Frage, Antwort und Zeit wird in einen MD5-Hash umgewandelt.
    """
    return hashlib.md5(f"{question}|{answer}|{timestamp}".encode("utf-8")).hexdigest()

def save_message_feedback(question, answer, rating):
    # Speichert das Feedback zu einer einzelnen Frage-Antwort-Kombination in die SQLite-Datenbank

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Aktueller Zeitstempel im Format "2024-05-18 14:32:00"
    try:
        conn = sqlite3.connect("message_feedback.db")
        cursor = conn.cursor()  

        # SQL-Befehl zum Einfügen der Daten (Frage, Antwort, Bewertung, Zeit)
        cursor.execute("""
            INSERT INTO message_feedback (timestamp, question, answer, rating)
            VALUES (?, ?, ?, ?)
        """, (timestamp, question, answer, rating))

        conn.commit()  # Änderungen speichern
        conn.close()   # Verbindung schließen
    except Exception as e:
        # Falls ein Fehler auftritt, wird dieser ins Log geschrieben
        logging.error(f"Fehler beim Speichern des Einzel-Feedbacks: {e}")

# 🚀 Initialisierung von Sitzungsvariablen (Session State)
# Diese werden nur einmal beim Start gesetzt – wenn sie noch nicht existieren

# Bot-Instanz nur einmal initialisieren (Konfiguration + Kernlogik laden)
if "bot" not in st.session_state:
    config = AppConfig()                    # lädt alle Systempfade und Modelleinstellungen
    st.session_state.bot = WittyBotCore(config)  # erstellt die zentrale Bot-Instanz

# Nachrichtenverlauf (Chat) initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []          # speichert alle bisherigen Nachrichten

# Zustand: Ob der Bot aktuell „nachdenkt“ (Antwort generiert)
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

# Zustand: Wurde der „Antwort abbrechen“-Button gedrückt?
if "abort_triggered" not in st.session_state:
    st.session_state.abort_triggered = False

# Zustand: Ist aktuell Feedback erlaubt für die letzte Antwort?
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False

# Zustand: Admin hat eine falsche Antwort erkannt und will korrigieren
if "feedback_incorrect" not in st.session_state:
    st.session_state.feedback_incorrect = False

# Zustand: „Danke für Feedback“-Hinweis anzeigen?
if "feedback_thanks" not in st.session_state:
    st.session_state.feedback_thanks = False

# ⬇️ Weitere Feedback-bezogene Zustände

# Erneut gesetzt (doppelt, aber zur Sicherheit)
if "feedback_thanks" not in st.session_state:
    st.session_state.feedback_thanks = False

# Zeitpunkt der letzten Feedbackaktion (für spätere Zeitprüfungen)
if "feedback_time" not in st.session_state:
    st.session_state.feedback_time = None

# Pro Nachricht: Wurde schon Feedback gegeben? (Verhindert mehrfaches Bewerten)
if "feedback_given_per_message" not in st.session_state:
    st.session_state.feedback_given_per_message = {}


# 🖐️ Begrüßungsnachricht des Assistenten
# Wird nur angezeigt, wenn der Chat leer ist (also beim ersten Start)
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",  # Rolle: Bot
        "content": "Hallo, ich bin **WittyBot** 🤖 – dein Studienassistent! Wie kann ich dir helfen?",
        "time": datetime.now().strftime("%d.%m.%Y, %H:%M")  # Aktueller Zeitstempel
    })

# 📥 Eingabefeld für die Nutzerfrage
user_input = None

# Eingabefeld nur anzeigen, wenn Bot nicht gerade antwortet
if not st.session_state.is_thinking:
    # Nutzer kann hier eine Frage stellen
    user_input = st.chat_input("Was möchtest du wissen?")
else:
    # Eingabefeld ist deaktiviert, solange Antwort generiert wird
    st.chat_input("WittyBot antwortet gerade... Bitte warte ⏳", disabled=True)

# 📤 Wenn der Benutzer etwas eingegeben hat (eine neue Frage)
if user_input:
    st.session_state.latest_question = user_input  # Speichere die Frage
    st.session_state.current_time = datetime.now().strftime("%d.%m.%Y, %H:%M")  # Aktuelle Uhrzeit speichern
    st.session_state.is_thinking = True           # Setze Status: Bot antwortet
    st.session_state.aborted = False              # Antwort wurde noch nicht abgebrochen
    st.rerun()                                     # Streamlit wird neu gestartet → Anzeige wird aktualisiert

# 💬 Wenn der Bot gerade antwortet
if st.session_state.is_thinking and "latest_question" in st.session_state:
    user_input = st.session_state.latest_question
    current_time = st.session_state.current_time

    # 💬 Zeige die Frage des Nutzers im Chat
    with st.chat_message("user", avatar="img/user.png"):
        st.markdown(f"<div style='text-align: right'>{user_input}<div class='timestamp'>{current_time}</div></div>", unsafe_allow_html=True)

    # 🤖 Wenn keine Abbruch-Anfrage vorliegt
    if not st.session_state.abort_triggered:
        with st.chat_message("assistant", avatar="img/logo.png"):

            # 🧠 Text: Der Bot denkt nach...
            st.markdown("WittyBot denkt nach... 💭")

            # ❌ Abbrechen-Button anzeigen
            st.markdown('<div class="abort-button">', unsafe_allow_html=True)
            if st.button("✖️ Antwort abbrechen"):
                st.session_state.abort_triggered = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # 🌀 Animation (Lottie) Llama – zentriert unter dem Abbrechen-Button
        import json
        with open("img/loading.json", "r", encoding="utf-8") as f:
            lottie_json = json.dumps(json.load(f))

        components.html(f"""
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <div style="display: flex; justify-content: center; margin-top: 4px;">
                <lottie-player
                    autoplay
                    loop
                    mode="normal"
                    style="width: 200px; height: 200px;"
                    src='data:application/json;base64,{base64.b64encode(lottie_json.encode()).decode()}' >
                </lottie-player>
            </div>
        """, height=260)

        # 🧠 Der Bot generiert jetzt die Antwort
        response = st.session_state.bot.handle_query(user_input)

        # Wenn Antwort nicht abgebrochen wurde → Antwort anzeigen
        if not st.session_state.aborted:
            st.markdown(f"{response}<div class='timestamp'>{current_time}</div>", unsafe_allow_html=True)

        # ✅ Speichere Frage + Antwort im Verlauf
        if not st.session_state.aborted:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "time": current_time
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "time": current_time
            })
            st.session_state.latest_answer = response
            st.session_state.feedback_mode = True  # Feedback-Modus aktivieren (für Admin)
    
        # Zurücksetzen aller temporären Zustände
        st.session_state.is_thinking = False
        del st.session_state.latest_question
        del st.session_state.current_time
        st.rerun()
    
    # ❌ Wenn die Antwort abgebrochen wurde
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "time": current_time
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Antwort wurde abgebrochen.",
            "time": current_time
        })
        st.session_state.is_thinking = False
        st.session_state.aborted = True
        st.session_state.abort_triggered = False
    
        # Aufräumen der Variablen
        if "latest_question" in st.session_state:
            del st.session_state.latest_question
        if "current_time" in st.session_state:
            del st.session_state.current_time
    
        st.rerun()

# 📋 Funktion: Kopier-Button für Nachrichten
# Fügt einen kleinen Button ein, mit dem man Text in die Zwischenablage kopieren kann
def copy_button(text):
    components.html(f"""
        <button onclick="navigator.clipboard.writeText(`{text}`)"
                style="background: none; border: none; padding: 0; margin: 0; cursor: pointer; font-size: 16px;">
            📋
        </button>
    """, height=35)

# Chat anzeigen
for msg in st.session_state.messages:
    timestamp = f"<div class='timestamp'>{msg['time']}</div>" if "time" in msg else ""
    if msg["role"] == "user":
        with st.chat_message("user", avatar="img/user.png"):
            st.markdown(f"<div style='text-align: right'>{msg['content']}{timestamp}</div>", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="img/logo.png"):
            idx = st.session_state.messages.index(msg)
            escaped = msg["content"].replace('"', "&quot;").replace("'", "&#39;")
    
            question = ""
            answer = msg["content"]
            if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                question = st.session_state.messages[idx - 1]["content"]
    
            # 💡 Keine Bewertung für Begrüßung oder Abbruch
            if idx == 0 or msg["content"].strip() == "Antwort wurde abgebrochen.":
                st.markdown(f"""
                    <div style="position: relative;">
                        <div style="white-space: pre-wrap;">{msg['content']}</div>
                        <button onclick="navigator.clipboard.writeText(`{escaped}`)" 
                                title="Kopieren" 
                                style="position: absolute; top: 0; right: 0; background: none; border: none; cursor: pointer; font-size: 16px;">
                            📋
                        </button>
                        <div class='timestamp'>{msg['time']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Normale Antwort mit Bewertung
                col1, col2, col3, col4 = st.columns([10, 1, 1, 1])  # добавили отдельную колонку
                with col1:
                    st.markdown(f"""
                        <div style="position: relative;">
                            <div style="white-space: pre-wrap;">{msg['content']}</div>
                            <div class='timestamp'>{msg['time']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # 👍 / 👎 — только если не был дан фидбек
                feedback_key = generate_message_key(question, answer, msg["time"])
                if not st.session_state.feedback_given_per_message.get(feedback_key):
                    with col2:
                        if st.button("👍", key=f"thumbs_up_{idx}"):
                            save_message_feedback(question, answer, 1)
                            st.session_state.feedback_given_per_message[feedback_key] = True
                            st.session_state.feedback_thanks = True
                            st.rerun()
                
                    with col3:
                        if st.button("👎", key=f"thumbs_down_{idx}"):
                            save_message_feedback(question, answer, -1)
                            st.session_state.feedback_given_per_message[feedback_key] = True
                            st.session_state.feedback_thanks = True
                            st.rerun()
                
                # 📋 — всегда показываем
                with col4:
                    copy_button(escaped)

# ⏳ Diese Funktion sorgt dafür, dass Streamlit nach einer kurzen Verzögerung neu geladen wird.
# Sie wird z. B. genutzt, um kurz eine Erfolgsmeldung zu zeigen und dann die Oberfläche zu aktualisieren.
import threading

def trigger_delayed_rerun(delay_sec: float = 2):
    # Diese innere Funktion wartet x Sekunden und startet dann ein Re-Rendern (Neuladen)
    def rerun_later():
        import time
        time.sleep(delay_sec)     # Wartezeit in Sekunden
        st.experimental_rerun()   # Seite neu laden

    # Starte die Wartezeit in einem eigenen Thread (damit die Hauptanwendung nicht blockiert wird)
    threading.Thread(target=rerun_later).start()

# ✅ Wenn nach einer Nachricht eine Feedback-Dankeschön-Nachricht gezeigt werden soll:
if st.session_state.feedback_thanks:
    container = st.empty()  # Ein leerer Platzhalter für die Erfolgsmeldung
    container.success("✅ Danke für dein Feedback! 👍")  # Zeige Erfolgsmeldung

    time.sleep(2)  # Zeige sie für 2 Sekunden

    container.empty()  # Danach entfernen wir die Meldung wieder

    # Status zurücksetzen – damit die Nachricht nur einmal erscheint
    st.session_state.feedback_thanks = False
                    
# 🧑‍💼 Admin-Feedback-Modus
# Dieser Bereich ist nur sichtbar, wenn:
# – der Admin-Modus aktiviert ist UND
# – Feedback zur letzten Antwort aktiviert wurde
if st.session_state.feedback_mode and st.session_state.is_admin:
    st.markdown("### 🗳️ War die Antwort korrekt?")  # Überschrift für Admin-Feedback

    # Zwei Spalten für zwei Buttons: ✅ und ❌
    col1, col2 = st.columns(2)

    # ✅ Antwort war korrekt
    with col1:
        if st.button("✅ Antwort war korrekt"):
            st.success("Danke für dein Feedback! 🙌")  # Erfolgsmeldung
            st.session_state.feedback_mode = False    # Feedback-Modus deaktivieren
            st.rerun()  # Seite neu laden

    # ❌ Antwort war falsch
    with col2:
        if st.button("❌ Antwort war falsch"):
            st.session_state.feedback_incorrect = True  # Weiter zur Korrektureingabe

# Wenn Admin sagt, dass die Antwort falsch war → Textfeld für manuelle Korrektur anzeigen
if st.session_state.feedback_incorrect and st.session_state.is_admin:
    # Textfeld, um die korrekte Antwort einzugeben
    correct_answer = st.text_area("✍️ Bitte gib die richtige Antwort ein:", height=100)

    # 💾 Speichern-Button
    if st.button("💾 Antwort speichern"):
        # Ursprüngliche Frage (die zweitletzte Nachricht im Verlauf)
        original_question = st.session_state.messages[-2]["content"]

        # Manuell korrigierte Antwort wird zur Wissensdatenbank hinzugefügt
        st.session_state.bot.knowledge_base.add_entry(original_question, correct_answer)

        # Bestätigung anzeigen
        st.success("✅ Richtige Antwort wurde gespeichert!")

        # Feedback-Zustände zurücksetzen
        st.session_state.feedback_mode = False
        st.session_state.feedback_incorrect = False
        del st.session_state.latest_answer

        # Seite neu laden
        st.rerun()
