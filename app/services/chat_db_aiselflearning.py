import os
import json
import mysql.connector


def get_mysql_connection_aisl():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "entraining_chat"),
        autocommit=True,
    )


def ensure_chat_session_aisl(conn, chat_id, OCourse_no=None, ip_address=None, user_agent=None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_session_ai_self_learning
            (chat_id, OCourse_no, ip_address, user_agent)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            OCourse_no = VALUES(OCourse_no),
            ip_address = VALUES(ip_address),
            user_agent = VALUES(user_agent),
            updated_at = CURRENT_TIMESTAMP
    """, (chat_id, OCourse_no, ip_address, user_agent))
    cur.close()


def load_chat_state_aisl(conn, chat_id):
    from app.schemas_aiselflearning import ChatState_aiselflearning

    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT state_json
        FROM chat_session_ai_self_learning
        WHERE chat_id = %s
        LIMIT 1
    """, (chat_id,))
    row = cur.fetchone()
    cur.close()

    if not row or not row.get("state_json"):
        return ChatState_aiselflearning()

    try:
        return ChatState_aiselflearning(**json.loads(row["state_json"]))
    except Exception:
        return ChatState_aiselflearning()


def save_chat_state_aisl(conn, chat_id, state):
    cur = conn.cursor()
    cur.execute("""
        UPDATE chat_session_ai_self_learning
        SET state_json = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE chat_id = %s
    """, (json.dumps(state.model_dump(), ensure_ascii=False), chat_id))
    cur.close()


def insert_chat_message_aisl(conn, chat_id, sender_type, message_text):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_message_ai_self_learning
            (chat_id, sender_type, message_text)
        VALUES (%s, %s, %s)
    """, (chat_id, sender_type, message_text))
    cur.close()


def insert_request_log_aisl(conn, chat_id, OCourse_no=None, ip_address=None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_request_log_ai_self_learning
            (chat_id, OCourse_no, ip_address)
        VALUES (%s, %s, %s)
    """, (chat_id, OCourse_no, ip_address))
    cur.close()


def reset_chat_state_aisl(conn, chat_id):
    cur = conn.cursor()
    cur.execute("""
        UPDATE chat_session_ai_self_learning
        SET state_json = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE chat_id = %s
    """, (chat_id,))
    cur.execute("""
        DELETE FROM chat_message_ai_self_learning
        WHERE chat_id = %s
    """, (chat_id,))
    cur.close()