# ---------------------------
# 1. IMPORTS & KONFIGURATION
# ---------------------------
# Systembezogene Imports
import sys
import logging
import sqlite3
import threading
import numpy as np
from datetime import datetime
import unicodedata
import re

# LangChain Komponenten
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Für lokale LLMs via Ollama
from langchain_chroma import Chroma  # Vektor-Datenbank
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Textaufteilung
from langchain_core.prompts import ChatPromptTemplate  # Prompt-Vorlagen
from langchain_core.output_parsers import StrOutputParser  # Ausgabeformatierung
from langchain_core.runnables import RunnablePassthrough  # Datenweitergabe in Chains

# Pfadverarbeitung und Typen
from pathlib import Path
from typing import List, Optional

# ---------------------------
# KONFIGURATION
# ---------------------------
class PathManager:
    """Zentrale Verwaltung aller Dateipfade
    (Kann bei Bedarf für unterschiedliche Umgebungen angepasst werden)"""
    def __init__(self):
        # Basisverzeichnis des Skripts
        self.script_dir = Path(__file__).parent.resolve()
        
        # Konfigurierbare Verzeichnisse:
        self.docs_dir = self.script_dir / "data"        # Dokumentenspeicher
        self.vector_db = self.script_dir / "chroma_db"  # Vektor-Datenbank
        self.sql_db = self.script_dir / "knowledge.db"  # SQLite Wissensdatenbank
        self.prompts_dir = self.script_dir / "prompt"   # Prompt-Vorlagen

        # Automatische Verzeichniserstellung
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

class AppConfig:
    """Hauptkonfiguration der Anwendung
    (Hier können zentrale Einstellungen angepasst werden)"""
    def __init__(self):
        self.paths = PathManager()
        
        # Embedding-Modell (Optionen: nomic-embed-text, jina/jina-embeddings-v2-base-de)
        self.embedding_params = {
            'model': "jina/jina-embeddings-v2-base-de",  # Modellname
            'chunk_size': 512,      # Context Windows
            'temperature': 0.0,     # Embedding-Temp (0.0-1.0)
            'top_k': 50             # Top-K Sampling
        }

        self.embeddings = OllamaEmbeddings(
            model=self.embedding_params['model'],
            temperature=self.embedding_params['temperature'],
            top_k=self.embedding_params['top_k']
        )
        
        # LLM-Modell (Optionen: mistral,(7b) llama3.1:8b, gemma3,(4B) qwen2.5 (7b), deepseek-r1:7b )
        # Für Leistungsschwache Systeme: (Optionen: qwen2.5:1.5b, deepseek-r1:1.5b, gemma3:1b, llama3.2:1b  )

        self.llm_params = {
            'model': "mistral:7b",
            'temperature': 0.1,     # Kreativität (0.0-1.0)
            'top_p': 0.85,         # Fokusbereich (0.0-1.0)
            'top_k': 20,            # Vokabularbeschränkung
            'num_ctx': 2048,        # Kontextfenster
            'repeat_penalty': 1.25, # Wiederholungsvermeidung
        }
        
        # Textverarbeitungsparameter
        self.text_processing = {
            'chunk_size': 256, # Größe der Textabschnitte in Zeichen
            'chunk_overlap': 64, # Überlappung zwischen Abschnitten
            'separators': ["\n\n", "\n", ". ", "! ", "? "]  # Split-Logik
        }


# ---------------------------
# 2. WISSENSDATENBANK
# ---------------------------
class ThreadSafeDB:
    """Thread-sichere SQLite Datenbankoperationen"""
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.execute_with_lock(
            """CREATE TABLE IF NOT EXISTS qna (
                id INTEGER PRIMARY KEY,
                question TEXT UNIQUE,  -- Eindeutige Fragen
                answer TEXT,
                source TEXT DEFAULT 'manual',  -- Quelle der Antwort
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        ):
            pass
        with self.execute_with_lock(
            """CREATE TABLE IF NOT EXISTS document_meta (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                last_modified REAL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        ):
            pass


    def execute_with_lock(self, query: str, params=()):
        """Führt eine Datenbankoperation mit Lock aus"""
        return DBContextManager(self.conn, self.lock, query, params)

class DBContextManager:
    """Sicherer Kontextmanager für Datenbanktransaktionen
    (Stellt sicher, dass Verbindungen korrekt geschlossen werden)"""
    def __init__(self, conn, lock, query, params):
        self.conn = conn
        self.lock = lock
        self.query = query
        self.params = params

    def __enter__(self):
        self.lock.acquire()
        self.cursor = self.conn.execute(self.query, self.params)
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.lock.release()

class KnowledgeBase:
    """Zentrale Schnittstelle für Wissensverwaltung
    (Kombiniert manuelle Einträge und Dokumentenwissen)"""
    def __init__(self, config: AppConfig):
        self.db = ThreadSafeDB(config.paths.sql_db)
        
    def add_entry(self, question: str, answer: str):
        """Fügt neuen manuellen Eintrag hinzu
        (Überschreibt vorhandene Einträge bei doppelten Fragen)"""
        with self.db.execute_with_lock(
            "INSERT OR REPLACE INTO qna (question, answer) VALUES (?, ?)",
            (question, answer)
        ):
            logging.info(f"💾 Neuer Eintrag: {question}")

    def get_answer(self, question: str) -> Optional[str]:
        """Sucht nach vorhandenen manuellen Antworten
        (Gibt None zurück wenn keine Antwort vorhanden)"""
        with self.db.execute_with_lock(
            "SELECT answer FROM qna WHERE question = ?",
            (question,)
        ) as cursor:
            result = cursor.fetchone()
            return result[0] if result else None

# ---------------------------
# 3. DOKUMENTENVERARBEITUNG
# ---------------------------
class DocumentProcessor:
    """Lädt und verarbeitet Dokumente für die Vektor-Datenbank
    (Unterstützt PDF und CSV, kann um weitere Formate erweitert werden)"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.embeddings = config.embeddings

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.text_processing['chunk_size'],
            chunk_overlap=config.text_processing['chunk_overlap'],
            separators=config.text_processing['separators']
        )

    
    def get_changed_files(self) -> tuple[list[Path], list[str]]:
        """Prüft alle unterstützten Dateien im Dokumentenordner auf Änderungen.
        Gibt zwei Listen zurück:
        - changed_files: Dateien, die neu hinzugefügt oder inhaltlich geändert wurden (basierend auf dem Änderungsdatum).
        - deleted_files: Dateien, die nicht mehr im Dateisystem existieren, aber noch in der Datenbank eingetragen sind.
        """
        changed_files = []
        all_files = [
            f for f in self.config.paths.docs_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in ['.pdf', '.csv']
        ]

        with self.knowledge_base.db.execute_with_lock(
            "SELECT file_path, last_modified FROM document_meta"
        ) as cursor:
            existing_files = {row[0]: row[1] for row in cursor.fetchall()}

        for file_path in all_files:
            db_mtime = existing_files.get(str(file_path))
            current_mtime = file_path.stat().st_mtime
            if not db_mtime or current_mtime > db_mtime:
                changed_files.append(file_path)

        deleted_files = set(existing_files.keys()) - {str(f) for f in all_files}
        return changed_files, list(deleted_files)

    def update_file_metadata(self, file_path: Path):
        """Speichert das aktuelle Änderungsdatum der Datei in der Datenbank.
        Wird nach erfolgreicher Verarbeitung aufgerufen, um spätere Änderungen erkennen zu können.
        """
        mtime = file_path.stat().st_mtime
        with self.knowledge_base.db.execute_with_lock(
            """INSERT OR REPLACE INTO document_meta (file_path, last_modified) VALUES (?, ?)""",
            (str(file_path), mtime)
        ):
            pass

    def remove_deleted_files(self, deleted_files: list[str]):
        """Entfernt gelöschte Dateien aus der Metadaten-Tabelle und der Vektor-Datenbank.
        Diese Methode wird aufgerufen, wenn Dateien in der Dateistruktur nicht mehr existieren.
        """
        for file_path in deleted_files:
            with self.knowledge_base.db.execute_with_lock(
                "DELETE FROM document_meta WHERE file_path = ?",
                (file_path,)
            ):
                pass
            self.vector_db._collection.delete(where={"source": file_path})


    def load_documents(self) -> List[Document]:
        """Dokumentenladevorgang mit Fehlerbehandlung
        (Ignoriert leere Dateien und nicht unterstützte Formate)"""
        loaders = [
            (PyPDFLoader, "*.pdf", "📄 PDF-Datei"),  # PDF-Verarbeitung
            (CSVLoader, "*.csv", "📊 CSV-Datei")     # CSV mit automatischer Encodungserkennung
        ]

        all_docs = []
        for loader_cls, pattern, icon in loaders:
            try:
                # Konfigurierbarer Loader mit Dateitypspezifischen Einstellungen
                loader = DirectoryLoader(
                    str(self.config.paths.docs_dir),
                    glob=pattern,
                    loader_cls=loader_cls,
                    loader_kwargs={'autodetect_encoding': True} if loader_cls == CSVLoader else {}
                )
                docs = loader.load()
                
                if not docs:
                    logging.warning(f"{icon} Keine {pattern}-Dateien gefunden")
                    continue
                
                # Textaufteilung für bessere Embeddingqualität
                split_docs = self.text_splitter.split_documents(docs)
                all_docs.extend(split_docs)
                logging.info(f"{icon} {len(split_docs)} Chunks aus {pattern}-Dateien")

            except Exception as e:
                logging.error(f"❌ Fehler bei {pattern}: {str(e)}")

        if not all_docs:
            raise ValueError("📭 Keine Dokumente gefunden! Bitte Dateien im data/-Ordner ablegen.")
        
        return all_docs
    

class TextNormalizer:
    """Diese Klasse sorgt für eine einheitliche Textbasis:
    - Kleinschreibung
    - Unicode-Normalisierung
    - Entfernung von Satzzeichen/Leerzeichen
    - Dient zur Verbesserung der semantischen Vergleichbarkeit in Suche & Cosine Checks.
    - Kann erweitet werden"""

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""

        # Unicode-Normalisierung (z. B. für Akzente)
        text = unicodedata.normalize("NFKC", text)

        # Kleinbuchstaben, Trimmen, Mehrfach-Leerzeichen entfernen
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)

        # Entferne optionale Satzzeichen am Ende (Fragezeichen etc.)
        text = re.sub(r"[?!.,;:\s]+$", "", text)

        return text
    

# ---------------------------
# 4. WITTYBOT KERN
# ---------------------------
class WittyBotCore:
    """Hauptlogik des Bots mit RAG-Pipeline
    (Retrieval Augmented Generation mit manuellen Antworten kombiniert)"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.knowledge_base = KnowledgeBase(config)
        self.vector_db = self.init_vector_db()  # Chroma-Datenbank
        self.chain = self.build_processing_chain()  # LangChain Verarbeitungskette
        self.conversation_context = []  # Dialogverlauf speichern
        self.max_context_length = 5     # Maximale Anzahl gespeicherter Nachrichten

    def init_vector_db(self):
        """Initialisiert die Vektordatenbank
        (Erstellt neue DB falls nicht vorhanden oder leer)"""
        processor = DocumentProcessor(self.config)
        
        # Neuerstellung bei erstem Start
        if not any(self.config.paths.vector_db.iterdir()):
            logging.info("🔨 Erstelle neue Vektor-DB...")
            return self.create_vector_db(processor.load_documents())
            
        db = Chroma(
            persist_directory=str(self.config.paths.vector_db),
            embedding_function=processor.embeddings
        )
        
        # Fallback bei Problemen
        if db._collection.count() == 0:
            logging.warning("⚠️ Leere DB - neuladen...")
            return self.create_vector_db(processor.load_documents())
            
        return db

    
    def reload_vector_db(self):
        processor = DocumentProcessor(self.config)
        processor.knowledge_base = self.knowledge_base
        processor.vector_db = self.vector_db

        changed_files, deleted_files = processor.get_changed_files()

        if not changed_files and not deleted_files:
            logging.info("✅ Keine Änderungen seit letztem Update")
            return

        if changed_files:
            logging.info(f"🔄 Verarbeite {len(changed_files)} geänderte Dateien")
            from langchain_community.document_loaders import PyPDFLoader, CSVLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            docs = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.text_processing['chunk_size'],
                chunk_overlap=self.config.text_processing['chunk_overlap'],
                separators=self.config.text_processing['separators']
            )

            for file in changed_files:
                loader_cls = PyPDFLoader if file.suffix.lower() == ".pdf" else CSVLoader
                loader_kwargs = {'autodetect_encoding': True} if file.suffix.lower() == ".csv" else {}
                loader = loader_cls(str(file), **loader_kwargs)
                raw_docs = loader.load()
                split_docs = splitter.split_documents(raw_docs)
                docs.extend(split_docs)

            self.vector_db.add_documents(docs)

            for f in changed_files:
                processor.update_file_metadata(f)

        if deleted_files:
            logging.info(f"🗑️ Entferne {len(deleted_files)} gelöschte Dateien")
            processor.remove_deleted_files(deleted_files)


    def create_vector_db(self, docs: List[Document]):
        """Erstellt eine neue Vektor-DB mit Dokumenten
        (Bereinigt vorhandene Daten vor dem Neuladen)"""
        for f in self.config.paths.vector_db.glob("*"):
            f.unlink()
            
        return Chroma.from_documents(
            documents=docs,
            embedding=self.config.embeddings,
            persist_directory=str(self.config.paths.vector_db)
        )

    def build_processing_chain(self):
        """Konfiguriert die LangChain Verarbeitungspipeline
        (Kombiniert Dokumentenkontext und manuelle Antworten)"""
        prompt_path = self.config.paths.script_dir / "prompt" / "main_prompt.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Zusammensetzung der Processing Chain
        return (
            {
                "documents": self.get_context,      # Dokumentenrecherche
                "manual_answers": self.get_manual_answer,  # SQL-Antworten
                "context": lambda x: "\n".join(self.conversation_context[-self.max_context_length*2:]),  # Kontext
                "question": RunnablePassthrough()   # Originalfrage weiterleiten
            }
            | prompt  # Prompt-Template anwenden
            | OllamaLLM(
                model=self.config.llm_params['model'],
                temperature=self.config.llm_params['temperature'],
                top_p=self.config.llm_params['top_p'],
                top_k=self.config.llm_params['top_k'],
                num_ctx=self.config.llm_params['num_ctx'],
                repeat_penalty=self.config.llm_params['repeat_penalty'],
            )
            | StrOutputParser()  # Antwortbereinigung
        )

    def update_context(self, question: str, answer: str):
        """Aktualisiert den Gesprächskontext"""
        self.conversation_context.extend([
            f"Frage: {question}",
            f"Antwort: {answer}"
        ])
        # Kürzen auf maximal erlaubte Länge
        self.conversation_context = self.conversation_context[-self.max_context_length*2:]

    def get_contextual_question(self, question: str) -> str:
        """ 
        - Kombiniert die aktuelle Frage mit dem letzten Frage-Antwort-Paar.
        - Entfernt dabei Quellenangaben aus der letzten Antwort,
        - um die semantische Suche nicht zu verfälschen.
        """
        if not self.conversation_context:
            return question  # kein Kontext, kein Umbau

        # Nur letzte Frage und Antwort (aber Antwort gekürzt oder bereinigt)
        last_q = self.conversation_context[-2]
        last_a = re.sub(r"\[Quelle:.*?\]", "", self.conversation_context[-1])  # Quelle raus

        return f"{last_q}\n{last_a}\nAktuelle Frage: {question}"

    def get_context(self, question: str) -> str:
        """Holt relevante Dokumentenausschnitte mittels Vector Similarity Search (mit dynamischem k und Score-Filter)"""
        contextual_query = self.get_contextual_question(question["question"])

        # 🔁 Dynamisches k abhängig von der Frage-Länge
        word_count = len(question["question"].split())

        if self.is_follow_up:
            k = 3
        else:
            if word_count <= 8:
                k  = 3
            elif word_count <= 15:
                k = 5
            else:
                k = 8

        logging.info(f"🔍 Suche mit dynamischem k={k} für Frage: '{question['question']}'")

        results = self.vector_db.similarity_search_with_score(contextual_query, k=k)

        threshold = 0.75
        filtered_docs = [doc for doc, score in results if score >= threshold]

        if not results:
            logging.warning("⚠️ Keine Dokumente gefunden.")
        elif not filtered_docs:
            logging.warning(f"⚠️ {len(results)} Dokumente gefunden, aber alle unter Score-Schwelle ({threshold}).")
        else:
            logging.info(f"📄 {len(filtered_docs)} relevante Dokumente gefunden für: '{question['question']}'")
            for doc, score in results:
                if score >= threshold:
                    logging.info(f"➡️ Quelle: {Path(doc.metadata['source']).name} S.{doc.metadata.get('page', '?')} | Score: {score:.2f}")
                    logging.debug(f"📑 Inhalt: {doc.page_content[:250].replace(chr(10), ' ')}...")
                else:
                    logging.debug(f"⛔️ Ignoriert: {Path(doc.metadata['source']).name} S.{doc.metadata.get('page', '?')} | Score: {score:.2f}")

        return "\n".join(
            f"• {doc.page_content}\n  📌 Quelle: {Path(doc.metadata['source']).name} S.{doc.metadata.get('page', '?')}"
            for doc in filtered_docs
        )


    def get_manual_answer(self, question: str) -> str:
        """Integration der manuellen Wissensdatenbank
        (Dient als Prioritätsquelle für spezifische Antworten)"""
        answer = self.knowledge_base.get_answer(question["question"])
        if answer:
            logging.info(f"📚 Manuelle Antwort gefunden: '{answer[:100]}...'")
        else:
            logging.info("ℹ️ Keine manuelle Antwort gefunden.")
        return answer or ""

    def handle_query(self, question: str) -> str:
        """ Zentrale Methode zur Verarbeitung einer Nutzeranfrage:"""

        original_question = question.strip()
        normalized_question = TextNormalizer.normalize(original_question)

        # 1. Folgefrage prüfen (auf Basis der normalisierten Eingabe)
        is_followup = self.is_follow_up(normalized_question)
        if is_followup:
            logging.info("↪️ Folgefrage erkannt – Kontext wird eingebunden.")
            query_input = self.get_contextual_question(original_question)
        else:
            query_input = original_question

        # 2. Manuelle Antwort suchen (am besten auf Basis der Originalfrage)
        manual = self.knowledge_base.get_answer(query_input)
        if manual:
            logging.info(f"📚 Manuelle Antwort gefunden für: '{original_question}'")
            return f"📚 Manuelle Antwort:\n{manual}"

        # 3. LLM-Antwort generieren
        logging.info(f"🔍 Starte Antwortgenerierung für Frage: '{original_question}'")
        response = self.chain.invoke({"question": query_input})

        # 4. Cosine-Similarity checken (normalisiert!)
        if not self.cosine_similarity_check(normalized_question, response):
            logging.warning("⚠️ Cosine-Similarity zu niedrig – prüfe Antwort mit LLM-Fallback.")
            if not self.is_answer_relevant_to_question(original_question, response):
                logging.warning("⚠️ Antwort wurde wegen Themenabweichung verworfen")
                return "❌ Die Antwort war nicht thematisch passend zur Frage. Dazu liegen mir keine Informationen vor."

        # 5. Kontext aktualisieren
        self.update_context(original_question, response)

        # 6. Optional: manuelle Pflege
        if "keine Informationen" in response:
            if result := self.handle_missing_knowledge(original_question):
                return result

        return response

    def handle_missing_knowledge(self, question: str) -> Optional[str]:
        """Benutzerinteraktion bei unbekannten Fragen
        (Ermöglicht manuelle Nachpflege des Wissens)"""

        if input("\n💡 Antwort fehlt! Manuell hinzufügen? (J/N): ").lower() == "j":
            answer = input("📝 Korrekte Antwort eingeben: ")
            self.knowledge_base.add_entry(question, answer)
            return "✅ Antwort gespeichert!"
        return None
    
    def is_answer_relevant_to_question(self, question: str, answer: str) -> bool:
        """Validiert, ob eine generierte Antwort thematisch zur Frage passt"""
        
        prompt_path = self.config.paths.script_dir / "prompt" / "relevance_prompt.txt"
        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        relevance_prompt = template.replace("{{ question }}", question).replace("{{ answer }}", answer)

        critic = OllamaLLM(
            model="mistral:7b",  # oder ein kleineres Modell wie qwen2.5:1.5b
            temperature=0.0
        )
        result = critic.invoke(relevance_prompt).strip().lower()
        return "ja" in result
    
    def cosine_similarity_check(self, question: str, answer: str) -> bool:
        """Führt einen semantischen Ähnlichkeitsvergleich zwischen Frage und Antwort durch."""

        # Vorab-Normalisierung sorgt für Robustheit gegenüber Groß-/Kleinschreibung, Satzzeichen etc.
        question = TextNormalizer.normalize(question)
        answer = TextNormalizer.normalize(answer)

        word_count = len(question.split())
        if word_count <= 8:
            threshold = 0.5
        elif word_count <= 15:
            threshold = 0.6
        else:
            threshold = 0.7

        vectors = self.config.embeddings.embed_documents([question, answer])

        vec_q = np.array(vectors[0])
        vec_a = np.array(vectors[1])
        cosine = np.dot(vec_q, vec_a) / (np.linalg.norm(vec_q) * np.linalg.norm(vec_a))

        logging.info(f"📐 Cosine Similarity: {cosine:.2f} (Schwelle: {threshold:.2f})")
        return cosine >= threshold

    
    def is_follow_up(self, question: str) -> bool:
        """Erkennt Folgefragen auf Basis des letzten Kontexts (Frage + Antwort)."""

        if not self.conversation_context:
            return False  # Ohne echten Kontext keine Folgefrage

        previous_context = "\n".join(self.conversation_context[-2:])

        prompt_path = self.config.paths.script_dir / "prompt" / "followup_prompt.txt"
        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        
        followup_prompt = template.replace("{{ context }}", previous_context).replace("{{ question }}", question)


        llm = OllamaLLM(model="mistral:7b", temperature=0.0)
        result = llm.invoke(followup_prompt).strip().lower()

        return "ja" in result


# ---------------------------
# 5. INTERAKTIVE SHELL
# ---------------------------
class InteractiveShell:
    """Einfache Kommandozeilenschnittstelle
    (Kann durch GUI oder API ersetzt werden)"""
    def __init__(self, bot: WittyBotCore):
        self.bot = bot
        self.commands = {
            "/exit": self.exit,     # Programmende
            "/add": self.add_entry, # Manueller Eintrag
            "/reload": self.reload_db,  # DB-Neuladen
            "/clearcontext": self.clear_context,  # Kontext löschen
            "/status": self.print_status  # Systemstatus
        }

    def start(self):
        """Hauptschleife der Benutzerinteraktion"""
        print("🔧 Befehle: /exit, /add, /reload, /clearcontext, /status")
        print("\n🤖 Witty Ready - Los geht's!")
        
        while True:
            try:
                query = input("\n🎓 Ihre Frage: ").strip()
                if not query:
                    continue
                    
                if cmd := self.commands.get(query.lower()):
                    cmd()
                else:
                    response = self.bot.handle_query(query)
                    print(f"\n🤖 Antwort:\n{response}")
                    
                    # Feedback nur bei generierten Antworten
                    if not response.startswith("📚 Manuelle Antwort"):
                        self._validate_response(query, response)
                    
            except KeyboardInterrupt:
                print("\n⏹️  Abgebrochen")

    def exit(self):
        """Beendet die Anwendung sauber"""
        print("\n👋 Auf Wiedersehen!")
        sys.exit(0)

    def _validate_response(self, question: str, response: str):
        """Fragt den Benutzer nach Feedback zur Antwort"""
        feedback = input("\n🗳️ Antwort bewerten (R=richtig/F=falsch/Enter=Überspringen): ").lower()
    
        if feedback == 'f':
            correct_answer = input("❌ Antwort ist falsch. Bitte korrekte Antwort eingeben: ")
            self.bot.knowledge_base.add_entry(question, correct_answer)
            print("✅ Korrekte Antwort wurde gespeichert!")
        elif feedback == 'r':
            print("👍 Danke für die Bestätigung!")
    

    def add_entry(self):
        """Manuelles Hinzufügen von Q&A-Paaren
        (Umgeht die automatische Verarbeitung)"""
        question = input("❓ Neue Frage: ")
        answer = input("💡 Antwort: ")
        self.bot.knowledge_base.add_entry(question, answer)
        print("✅ Eintrag gespeichert")

    def reload_db(self):
        """Erzwingt Neuladen der Vektor-Datenbank
        (Nützlich bei Dokumentenänderungen)"""
        self.bot.reload_vector_db()
        print("🔄 Vektor-Datenbank inkrementell aktualisiert")

    def clear_context(self):
        """Löscht den Gesprächsverlauf"""
        self.bot.conversation_context = []
        print("🧹 Dialogkontext zurückgesetzt")

    def print_status(self):
        """Zeigt aktuelle Systeminformationen"""
    
        num_qna = self._count_entries("qna")
        num_docs = self._count_entries("document_meta")
        context_len = len(self.bot.conversation_context) // 2

        vector_db_path = self.bot.config.paths.vector_db
        db_files = list(vector_db_path.glob("*"))
        vector_db_size = sum(f.stat().st_size for f in db_files) / 1024

        print(f"""
📊 Systemstatus:
• Einträge in Wissensdatenbank (Q&A): {num_qna}
• Dokumente in Vektor-DB-Index:       {num_docs}
• Aktueller Gesprächskontext:         {context_len} Fragen
• Größe der Vektor-DB (persist):      {vector_db_size:.1f} KB
• Letztes Prompt-Update (main):       {self._get_prompt_mtime("main_prompt.txt")}
• Letztes Prompt-Update (followup):   {self._get_prompt_mtime("followup_prompt.txt")}
• Letztes Prompt-Update (relevance):  {self._get_prompt_mtime("relevance_prompt.txt")}
""")

    def _count_entries(self, table_name: str) -> int:
        with self.bot.knowledge_base.db.execute_with_lock(f"SELECT COUNT(*) FROM {table_name}") as cursor:
            return cursor.fetchone()[0]

    def _get_prompt_mtime(self, filename: str) -> str:
        try:
            path = self.bot.config.paths.prompts_dir / filename
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            return mtime.strftime("%Y-%m-%d %H:%M")
        except:
            return "❌ nicht gefunden"

# ---------------------------
# HAUPTPROGRAMM
# ---------------------------
if __name__ == "__main__":
    # Logger-Konfiguration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    try:
        config = AppConfig()  # Lade Konfiguration
        bot = WittyBotCore(config)  # Initialisiere Bot-Kern
        shell = InteractiveShell(bot)  # Starte Interface
        
        # Systemstatusausgabe
        print(f"""
🌐 WittyBot Systemstatus:
📂 Dokumentenordner: {config.paths.docs_dir}
🧠 Vektor-DB: {config.paths.vector_db}
💾 Wissensdatenbank: {config.paths.sql_db}
📑 Prompt-Vorlagen: {config.paths.script_dir}
        """)
        
        shell.start()
        
    except Exception as e:
        logging.error(f"❌ Kritischer Fehler: {str(e)}")
        sys.exit(1)