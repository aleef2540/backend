from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import psycopg2
import psycopg2.extras
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def get_course_by_no(course_no: int) -> Optional[Dict[str, Any]]:
    """
    ดึงข้อมูล course หลักจาก DB
    ปรับ SQL ตาม table จริงของคุณได้เลย
    """
    sql = """
        SELECT
            ocourse_no,
            ocourse_nameth,
            ocourse_nameen,
            ocourse_urlpicture,
            ocourse_key,
            ocourse_platform,
            ocourse_saleflag
        FROM course_online
        WHERE ocourse_no = %s
        LIMIT 1
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (course_no,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_course_videos(course_no: int) -> List[Dict[str, Any]]:
    """
    ดึงรายการวิดีโอของ course
    """
    sql = """
        SELECT
            video_part,
            video_name,
            video_duration,
            embed_youtube
        FROM course_online_vdo
        WHERE video_ocourse_no = %s
          AND video_type = 'VDO'
          AND COALESCE(video_name, '') <> ''
        ORDER BY video_part ASC
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (course_no,))
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]
    finally:
        conn.close()


def get_course_script_chunks(course_no: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    ตรงนี้เป็นตัวอย่างสำหรับดึง script / knowledge ของ course
    คุณเปลี่ยน table จริงทีหลังได้
    ถ้ายังไม่มี table script ให้ return [] ไปก่อน
    """
    sql = """
        SELECT
            id,
            course_no,
            part_no,
            chunk_text
        FROM ai_data_sl
        WHERE course_no = %s
        ORDER BY part_no ASC, id ASC
        LIMIT %s
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(sql, (course_no, limit))
                rows = cur.fetchall() or []
                return [dict(r) for r in rows]
            except Exception:
                # กันกรณี table ยังไม่พร้อม
                conn.rollback()
                return []
    finally:
        conn.close()


def build_course_context(
    course: Dict[str, Any],
    videos: List[Dict[str, Any]],
    script_chunks: List[Dict[str, Any]],
) -> str:
    """
    รวม context ให้ AI อ่าน
    """
    name_th = course.get("ocourse_nameth") or ""
    name_en = course.get("ocourse_nameen") or ""

    lines: List[str] = []
    lines.append(f"ชื่อหลักสูตร: {name_th}")
    if name_en:
        lines.append(f"ชื่อภาษาอังกฤษ: {name_en}")

    if videos:
        lines.append("รายการวิดีโอในหลักสูตร:")
        for v in videos[:15]:
            lines.append(
                f"- Part {v.get('video_part')}: {v.get('video_name')} "
                f"(duration={v.get('video_duration')})"
            )

    if script_chunks:
        lines.append("เนื้อหาสำคัญของหลักสูตร:")
        for chunk in script_chunks[:20]:
            part_no = chunk.get("part_no")
            chunk_text = (chunk.get("chunk_text") or "").strip()
            if chunk_text:
                lines.append(f"[Part {part_no}] {chunk_text}")

    return "\n".join(lines).strip()


def ask_ai_with_course_context(user_message: str, context: str) -> str:
    """
    เรียก AI ให้ตอบจากบริบทของหลักสูตร
    """
    system_prompt = """
คุณคือ AI ผู้ช่วยตอบคำถามบนหน้า Self Learning
หน้าที่ของคุณคือช่วยอธิบาย ตอบคำถาม สรุป และเชื่อมโยงสิ่งที่ผู้เรียนถาม
โดยอ้างอิงจากข้อมูลหลักสูตรที่ได้รับเป็นหลัก

กติกา:
- ตอบเป็นภาษาไทย
- ตอบให้เข้าใจง่าย กระชับ แต่มีสาระ
- ถ้าคำถามตอบได้จากข้อมูลหลักสูตร ให้ตอบจากข้อมูลนั้นก่อน
- ถ้าข้อมูลไม่พอ ให้บอกตรง ๆ ว่าในข้อมูลหลักสูตรที่มี ยังไม่พบรายละเอียดชัดเจน
- ห้ามแต่งข้อมูลเฉพาะเจาะจงที่ไม่มีใน context
- ถ้าเหมาะสม สามารถสรุปเป็นข้อสั้น ๆ ได้
""".strip()

    user_prompt = f"""
ข้อมูลหลักสูตร:
{context}

คำถามผู้ใช้:
{user_message}
""".strip()

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def answer_self_learning_question(
    chat_id: str,
    course_no: int,
    user_message: str,
) -> Dict[str, Any]:
    course = get_course_by_no(course_no)
    if not course:
        return {
            "ok": False,
            "reply": "ขออภัยครับ ไม่พบข้อมูลหลักสูตรที่ต้องการ",
            "course": None,
            "sources": [],
            "debug": {
                "chat_id": chat_id,
                "OCourse_no": course_no,
                "reason": "course_not_found",
            },
        }

    videos = get_course_videos(course_no)
    script_chunks = get_course_script_chunks(course_no, limit=20)

    context = build_course_context(
        course=course,
        videos=videos,
        script_chunks=script_chunks,
    )

    if not context.strip():
        return {
            "ok": True,
            "reply": "ขณะนี้พบข้อมูลหลักสูตรเบื้องต้น แต่ยังไม่มีรายละเอียดเพียงพอสำหรับตอบคำถามเชิงลึกครับ",
            "course": {
                "OCourse_no": course.get("ocourse_no"),
                "OCourse_nameTH": course.get("ocourse_nameth"),
                "OCourse_nameEN": course.get("ocourse_nameen"),
            },
            "sources": [],
            "debug": {
                "chat_id": chat_id,
                "OCourse_no": course_no,
                "reason": "empty_context",
            },
        }

    reply = ask_ai_with_course_context(
        user_message=user_message,
        context=context,
    )

    return {
        "ok": True,
        "reply": reply,
        "course": {
            "OCourse_no": course.get("ocourse_no"),
            "OCourse_nameTH": course.get("ocourse_nameth"),
            "OCourse_nameEN": course.get("ocourse_nameen"),
        },
        "sources": [
            {
                "type": "course_online",
                "OCourse_no": course.get("ocourse_no"),
            },
            {
                "type": "course_online_vdo",
                "count": len(videos),
            },
            {
                "type": "ai_data_sl",
                "count": len(script_chunks),
            },
        ],
        "debug": {
            "chat_id": chat_id,
            "OCourse_no": course_no,
            "video_count": len(videos),
            "script_chunk_count": len(script_chunks),
        },
    }