"""
AABS Control Tower - Memory System

SQLite-backed memory system for tracking recommendations, actions, and outcomes.
Provides learning loop capabilities by recording what the AI recommends,
what actions users take, and whether those actions were successful.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


class MemorySystem:
    """
    Memory system for tracking recommendations, actions, and outcomes.
    SQLite backend for performance, ACID compliance, and concurrent access.
    """

    def __init__(self):
        self.memory_dir = Path('memory')
        self.memory_dir.mkdir(exist_ok=True)
        self.db_path = self.memory_dir / 'control_tower.db'
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT, context TEXT,
                    timestamp TEXT NOT NULL, status TEXT DEFAULT 'pending',
                    action_timestamp TEXT, action_notes TEXT, outcome TEXT,
                    outcome_details TEXT, outcome_timestamp TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rec_timestamp ON recommendations(timestamp)')
            cursor.execute('CREATE TABLE IF NOT EXISTS metrics (key TEXT PRIMARY KEY, value REAL, updated_at TEXT)')
            cursor.execute('SELECT COUNT(*) FROM metrics')
            if cursor.fetchone()[0] == 0:
                default_metrics = [
                    ('total_recommendations', 0), ('acted_on', 0), ('ignored', 0),
                    ('modified', 0), ('successful_outcomes', 0), ('failed_outcomes', 0),
                    ('pending_outcomes', 0), ('trust_score', 0.5)
                ]
                cursor.executemany('INSERT INTO metrics (key, value, updated_at) VALUES (?, ?, ?)',
                                  [(k, v, datetime.now().isoformat()) for k, v in default_metrics])

    def log_recommendation(self, rec_type: str, content: dict, context: dict = None) -> str:
        rec_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO recommendations (id, type, content, context, timestamp, status) VALUES (?, ?, ?, ?, ?, "pending")',
                          (rec_id, rec_type, json.dumps(content, default=str), json.dumps(context or {}, default=str), timestamp))
            cursor.execute('UPDATE metrics SET value = value + 1, updated_at = ? WHERE key = ?', (timestamp, 'total_recommendations'))
        return rec_id

    def record_action(self, rec_id: str, action: str, notes: str = None):
        timestamp = datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE recommendations SET status = ?, action_timestamp = ?, action_notes = ? WHERE id = ?', (action, timestamp, notes, rec_id))
            metric_key = 'acted_on' if action == 'acted' else action
            cursor.execute('UPDATE metrics SET value = value + 1, updated_at = ? WHERE key = ?', (timestamp, metric_key))

    def record_outcome(self, rec_id: str, outcome: str, details: dict = None):
        timestamp = datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE recommendations SET outcome = ?, outcome_details = ?, outcome_timestamp = ? WHERE id = ?',
                          (outcome, json.dumps(details or {}, default=str), timestamp, rec_id))
            if outcome == 'success':
                cursor.execute('UPDATE metrics SET value = value + 1, updated_at = ? WHERE key = ?', (timestamp, 'successful_outcomes'))
            elif outcome == 'failed':
                cursor.execute('UPDATE metrics SET value = value + 1, updated_at = ? WHERE key = ?', (timestamp, 'failed_outcomes'))
            self._recalculate_trust_score(cursor, timestamp)

    def _recalculate_trust_score(self, cursor, timestamp):
        cursor.execute('SELECT value FROM metrics WHERE key = "successful_outcomes"')
        success = cursor.fetchone()[0]
        cursor.execute('SELECT value FROM metrics WHERE key = "failed_outcomes"')
        failed = cursor.fetchone()[0]
        total = success + failed
        if total > 0:
            trust_score = round(success / total, 3)
            cursor.execute('UPDATE metrics SET value = ?, updated_at = ? WHERE key = "trust_score"', (trust_score, timestamp))

    def get_metrics(self) -> dict:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM metrics')
            return {row['key']: row['value'] for row in cursor.fetchall()}

    def get_recent_recommendations(self, limit: int = 20) -> list:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT ?', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_outcomes(self) -> list:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recommendations WHERE status IN ('acted', 'modified') AND outcome IS NULL ORDER BY timestamp DESC LIMIT 20")
            return [dict(row) for row in cursor.fetchall()]

    def get_learning_insights(self) -> dict:
        m = self.get_metrics()
        total_recs = max(m.get('total_recommendations', 1), 1)
        total_outcomes = max(m.get('successful_outcomes', 0) + m.get('failed_outcomes', 0), 1)
        return {
            'trust_score': m.get('trust_score', 0.5),
            'total_recommendations': int(m.get('total_recommendations', 0)),
            'action_rate': round(((m.get('acted_on', 0) + m.get('modified', 0)) / total_recs) * 100, 1),
            'success_rate': round((m.get('successful_outcomes', 0) / total_outcomes) * 100, 1),
            'recommendations_today': 0
        }
