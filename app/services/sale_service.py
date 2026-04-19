from dotenv import load_dotenv
load_dotenv()

import json
import re
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient

from app.schemas_aisale import ChatState_aisale, IntentResult_aisale, SaleProgress_aisale
from app.services.call_ai import call_openai_chat_full, call_openai_embedding_full


client = OpenAI()

qdrant = QdrantClient(
    url="https://f9a8611d-3692-4b14-bd09-bfaa135fe05d.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTllMzhiNzAtNTgyOS00MDUyLTgxNzMtNzMwZTIzMjgxZTA2In0.ktGpvSyHUg_44H6ihEdwthPfY6OG_0qU3zxb4hT5FNQ",
)


def normalize_text(value: Any) -> str:
    if value is None:
        return "unknown"

    value = str(value).strip()
    if not value:
        return "unknown"

    lowered = value.lower()
    if lowered in {"unknown", "null", "none", "n/a", "-", "--"}:
        return "unknown"

    return value


def extract_json_object(text: str) -> dict:
    if not text or not str(text).strip():
        return {}

    cleaned = re.sub(r"```json|```", "", str(text), flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        possible_json = match.group(0)
        try:
            return json.loads(possible_json)
        except Exception:
            return {}

    return {}


async def detect_sale_intent(user_message: str) -> IntentResult_aisale:
    system_prompt = """
คุณคือ AI ที่มีหน้าที่จำแนก intent ของข้อความผู้ใช้ สำหรับระบบ AI Sale ของสถาบัน

ให้เลือกเพียง 1 ค่าเท่านั้น:
- greeting = ข้อความทักทายสั้น ๆ เช่น สวัสดี หวัดดี hello hi
- general = พูดคุยทั่วไป หรือยังไม่ชัดว่ากำลังถามหาหลักสูตร
- sale = ผู้ใช้กำลังมองหาหลักสูตร อยากพัฒนาเรื่องใดเรื่องหนึ่ง มีปัญหาในงาน หรืออยากให้แนะนำการอบรม

กฎสำคัญ:
- ถ้าผู้ใช้พูดถึงการพัฒนา การอบรม หลักสูตร คอร์ส ทักษะ ปัญหาในการทำงาน หรืออยากได้คำแนะนำเพื่อพัฒนาคน
  ให้เลือก sale
- ถ้าไม่แน่ใจแต่มีแนวโน้มว่าจะถามหาหลักสูตร ให้เลือก sale
- ถ้าเป็นแค่ทักทาย ให้ greeting
- ถ้ายังเป็น small talk ทั่วไป ให้ general

ตอบเป็น JSON เท่านั้น:
{
  "intent": "greeting|general|sale"
}
    """.strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_message,
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    decoded = extract_json_object(text)
    intent = decoded.get("intent", "general")

    if intent not in ["greeting", "general", "sale"]:
        intent = "general"

    return IntentResult_aisale(intent=intent)


async def reply_sale_greeting(user_message: str) -> str:
    system_prompt = """
คุณคือ AI Sale Assistant ของสถาบันฝึกอบรม
หน้าที่คือทักทายอย่างเป็นธรรมชาติ และชวนผู้ใช้เล่าว่ากำลังมองหาหลักสูตรหรืออยากพัฒนาเรื่องใด

ข้อกำหนด:
- ตอบ 1-2 ประโยค
- สุภาพ เป็นธรรมชาติ
- คุยง่าย
- ประโยคสุดท้ายควรเป็นคำถามเปิด
- ห้ามตอบยาว
    """.strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_message,
        temperature=0.6,
    )

    content = (result.get("content") or "").strip()
    if not content:
        content = "สวัสดีครับ ตอนนี้กำลังมองหาหลักสูตรหรืออยากพัฒนาด้านไหนเป็นพิเศษครับ"

    return content


async def reply_sale_general(user_message: str) -> str:
    system_prompt = """
คุณคือ AI Sale Assistant ของสถาบันฝึกอบรม
หน้าที่คือรับข้อความทั่วไป แล้วค่อย ๆ พาผู้ใช้เข้าสู่การเล่าความต้องการ เพื่อแนะนำหลักสูตรที่เหมาะ

ข้อกำหนด:
- ตอบ 1-2 ประโยค
- สุภาพ เป็นธรรมชาติ
- อย่าถามหลายคำถามในครั้งเดียว
- ปิดท้ายด้วยคำถามเปิดที่ชวนให้ผู้ใช้เล่าความต้องการ
    """.strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_message,
        temperature=0.6,
    )

    content = (result.get("content") or "").strip()
    if not content:
        content = "ได้เลยครับ ช่วยเล่าเพิ่มอีกนิดได้ไหมครับว่าตอนนี้อยากหาหลักสูตรเพื่อพัฒนาเรื่องอะไร"

    return content


async def analyze_sale_progress(
    user_message: str,
    state: ChatState_aisale,
    model: str = "gpt-4.1-mini",
) -> SaleProgress_aisale:
    current_topic = normalize_text(getattr(state, "topic", "unknown"))
    current_target_group = normalize_text(getattr(state, "target_group", "unknown"))
    current_goal = normalize_text(getattr(state, "goal", "unknown"))
    current_pain_point = normalize_text(getattr(state, "pain_point", "unknown"))
    last_question = str(getattr(state, "last_question", "none") or "none").strip()
    last_question_type = str(getattr(state, "last_question_type", "none") or "none").strip()

    system_prompt = f"""
คุณคือ AI ที่ทำหน้าที่สกัด requirement สำหรับแนะนำหลักสูตรจาก "ข้อความล่าสุดของผู้ใช้"
ให้แยกข้อมูลออกเป็น 4 ช่องคือ topic, target_group, goal, pain_point

=====================
บริบทปัจจุบัน
=====================
- topic ปัจจุบัน: {current_topic}
- target_group ปัจจุบัน: {current_target_group}
- goal ปัจจุบัน: {current_goal}
- pain_point ปัจจุบัน: {current_pain_point}
- last_question: {last_question}
- last_question_type: {last_question_type}

รอบนี้ให้วิเคราะห์โดยเน้น field เป้าหมายคือ: {last_question_type}
- ถ้าข้อความล่าสุดตอบ field นี้ได้ ให้เติม field นี้ก่อน
- field อื่นเติมได้เฉพาะเมื่อชัดเจนมาก
- ถ้าข้อความสั้น ให้ใช้บริบทจาก last_question ได้

=====================
ความหมายของแต่ละ field
=====================
1. topic
= หัวข้อหลักที่ลูกค้ากำลังอยากพัฒนา หรือหัวข้อหลักสูตรที่สนใจ
เช่น ภาวะผู้นำ การสื่อสาร การบริการ การขาย การทำงานเป็นทีม การบริหารเวลา

2. target_group
= กลุ่มผู้เข้าอบรม
เช่น พนักงานทั่วไป หัวหน้างาน ผู้จัดการ ผู้บริหาร ทีมขาย

3. goal
= เป้าหมายที่อยากให้เกิดขึ้นหลังพัฒนา
เช่น สื่อสารให้ชัดขึ้น บริหารทีมได้ดีขึ้น เพิ่มยอดขาย ทำงานร่วมกันได้ดีขึ้น

4. pain_point
= ปัญหาปัจจุบันที่กำลังเจอ
เช่น สั่งงานไม่ชัด ทีมไม่ค่อยร่วมมือ ขายไม่ถึงเป้า ลูกค้าไม่มั่นใจ

=====================
กฎการแยก field
=====================
- ข้อความเดียวอาจมีมากกว่า 1 field ได้
- ให้ดึงทุก field ที่พบ
- ถ้า field ไหนไม่ชัดจริง ๆ ให้ตอบ "unknown"
- ห้ามเดาข้อมูลใหม่เกินจากข้อความผู้ใช้
- อนุญาตให้สรุปถ้อยคำได้ โดยห้ามเพิ่มความหมายใหม่

=====================
กฎสำคัญเรื่อง topic
=====================
- topic ต้องเป็นหัวข้อหลัก ไม่ใช่ประโยคยาว
- ถ้ายังไม่ชัดจริง ๆ ให้ตอบ unknown

=====================
กฎสำคัญเรื่อง target_group
=====================
- ต้องเป็นกลุ่มผู้เข้าอบรม
- ถ้ายังไม่รู้ว่าผู้เข้าอบรมเป็นใคร ให้ตอบ unknown

=====================
กฎสำคัญเรื่อง goal
=====================
- goal คือสิ่งที่อยากให้ดีขึ้นหรือผลลัพธ์ที่อยากได้
- ถ้าเล่าปัญหาอย่างเดียว แต่ไม่ได้บอกผลลัพธ์ที่อยากได้ ให้ตอบ unknown

=====================
กฎสำคัญเรื่อง pain_point
=====================
- pain_point คือปัญหาที่กำลังเกิดขึ้นในปัจจุบัน
- ถ้าผู้ใช้ยังไม่ได้พูดถึงปัญหา ให้ตอบ unknown

=====================
ตัวอย่าง
=====================
ข้อความ: "อยากได้หลักสูตรพัฒนาหัวหน้างาน เพราะตอนนี้สั่งงานยังไม่ชัด"
ตอบ:
{{
  "topic": "การพัฒนาหัวหน้างาน",
  "target_group": "หัวหน้างาน",
  "goal": "unknown",
  "pain_point": "สั่งงานยังไม่ชัด"
}}

ข้อความ: "ทีมขายปิดการขายไม่ค่อยได้ อยากหาหลักสูตรช่วยเพิ่มยอดขาย"
ตอบ:
{{
  "topic": "การขาย",
  "target_group": "ทีมขาย",
  "goal": "เพิ่มยอดขาย",
  "pain_point": "ปิดการขายไม่ค่อยได้"
}}

ข้อความ: "อยากพัฒนาการสื่อสารในทีม"
ตอบ:
{{
  "topic": "การสื่อสาร",
  "target_group": "unknown",
  "goal": "พัฒนาการสื่อสารในทีม",
  "pain_point": "unknown"
}}

=====================
รูปแบบคำตอบ
=====================
ตอบเป็น JSON object เท่านั้น
ห้ามมีคำอธิบายอื่น

{{
  "topic": "string",
  "target_group": "string",
  "goal": "string",
  "pain_point": "string"
}}
    """.strip()

    result = await call_openai_chat_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=f"ข้อความล่าสุดของผู้ใช้: {user_message}",
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    decoded = extract_json_object(text)

    topic = normalize_text(decoded.get("topic", "unknown"))
    target_group = normalize_text(decoded.get("target_group", "unknown"))
    goal = normalize_text(decoded.get("goal", "unknown"))
    pain_point = normalize_text(decoded.get("pain_point", "unknown"))

    if topic == "unknown":
        topic = current_topic

    if target_group == "unknown":
        target_group = current_target_group

    if goal == "unknown":
        goal = current_goal

    if pain_point == "unknown":
        pain_point = current_pain_point

    if topic == "unknown":
        next_action = "ask_topic"
    elif target_group == "unknown":
        next_action = "ask_target_group"
    elif goal == "unknown":
        next_action = "ask_goal"
    elif pain_point == "unknown":
        next_action = "ask_pain_point"
    else:
        next_action = "ready"

    return SaleProgress_aisale(
        topic=topic,
        target_group=target_group,
        goal=goal,
        pain_point=pain_point,
        last_question=last_question,
        next_action=next_action,
        raw=text,
    )


async def generate_sale_question(
    user_message: str,
    state: ChatState_aisale,
    question_type: int,
    model: str = "gpt-4.1-mini",
) -> str:
    """
    question_type:
    1 = ask topic
    2 = ask target_group
    3 = ask goal
    4 = ask pain_point
    """

    topic = str(state.topic or "unknown").strip()
    target_group = str(state.target_group or "unknown").strip()
    goal_value = str(state.goal or "unknown").strip()
    pain_point = str(state.pain_point or "unknown").strip()
    last_question = str(state.last_question or "unknown").strip()

    if question_type == 1:
        missing_info = "ยังไม่รู้ว่าผู้ใช้อยากพัฒนาหรือสนใจหลักสูตรเรื่องอะไร"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุหัวข้อที่อยากพัฒนาให้ชัดขึ้น "
            "เช่น ภาวะผู้นำ การสื่อสาร การขาย การบริการ หรือการทำงานเป็นทีม"
        )
        question_focus = "topic"

    elif question_type == 2:
        missing_info = "ยังไม่รู้ว่ากลุ่มผู้เข้าอบรมเป็นใคร"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุกลุ่มเป้าหมายของผู้เข้าอบรม "
            "เช่น พนักงานทั่วไป หัวหน้างาน ผู้จัดการ ผู้บริหาร หรือทีมขาย"
        )
        question_focus = "target_group"

    elif question_type == 3:
        missing_info = "รู้หัวข้อและกลุ่มแล้ว แต่ยังไม่รู้ว่าอยากให้เกิดผลลัพธ์อะไร"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุสิ่งที่อยากให้ดีขึ้นหรือผลลัพธ์ที่ต้องการ "
            "เช่น สื่อสารให้ชัดขึ้น บริหารทีมดีขึ้น เพิ่มยอดขาย หรือทำงานร่วมกันดีขึ้น"
        )
        question_focus = "goal"

    elif question_type == 4:
        missing_info = "ยังไม่รู้ว่าปัญหาปัจจุบันที่กำลังเจอคืออะไร"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้เล่าปัญหาปัจจุบันที่กำลังเจอ "
            "เพื่อให้แนะนำหลักสูตรได้ตรงขึ้น"
        )
        question_focus = "pain_point"

    else:
        return "ช่วยเล่าเพิ่มเติมอีกนิดได้ไหมครับ"

    system_prompt = f"""คุณคือ AI Sale Assistant ของสถาบันฝึกอบรมที่กำลังคุยกับผู้ใช้อย่างต่อเนื่อง

หน้าที่:
- อ่านข้อความล่าสุดของผู้ใช้
- ดูว่าตอนนี้รู้อะไรแล้ว และยังขาดอะไร
- สร้าง “คำถามถัดไป” ที่ต่อเนื่องจากสิ่งที่ผู้ใช้เพิ่งพูด
- ถามเฉพาะข้อมูลที่ยังขาดอยู่เท่านั้น

ข้อมูลปัจจุบัน:
- topic: {topic}
- target_group: {target_group}
- goal: {goal_value}
- pain_point: {pain_point}
- สิ่งที่ยังขาด: {missing_info}
- เป้าหมายของคำถามนี้: {goal_instruction}
- focus ที่ต้องถาม: {question_focus}

หลักการถาม:
1. ถามเพียง 1 คำถามเท่านั้น
2. ห้ามถามหลายประเด็นในประโยคเดียว
3. ห้ามถามซ้ำสิ่งที่มีอยู่แล้ว
4. ให้ต่อเนื่องจากสิ่งที่ผู้ใช้เพิ่งพูด
5. ใช้ภาษาธรรมชาติ สุภาพ และช่วยให้ผู้ใช้ตอบง่าย
6. ห้ามทำให้เหมือนแบบฟอร์ม
7. สามารถยกตัวอย่างสั้น ๆ ได้ ถ้าช่วยให้ตอบง่ายขึ้น

รูปแบบคำตอบ:
- ตอบเป็นภาษาไทย
- 1 ประโยค หรือไม่เกิน 2 ประโยค
- ตอบเป็นข้อความคำถามเท่านั้น
    """

    user_prompt = f"""คำถามล่าสุดของผู้ช่วย: {last_question}
ข้อความล่าสุดของผู้ใช้: {user_message}

ช่วยสร้างคำถามถัดไปที่ถามเฉพาะข้อมูลที่ยังขาด"""

    result = await call_openai_chat_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
    )

    content = (result.get("content") or "").strip()

    if not content:
        content = "ช่วยเล่าเพิ่มเติมอีกนิดได้ไหมครับ เพื่อให้ผมช่วยแนะนำหลักสูตรได้ตรงขึ้น"

    return content


def build_sale_query_text_by_ai(
    topic: str,
    target_group: str,
    goal: str,
    pain_point: str,
) -> dict:
    system = """
คุณคือ AI ที่ช่วยสร้าง query text สำหรับค้นหาหลักสูตรใน vector database ของสถาบัน

หน้าที่:
- รับ topic, target_group, goal, pain_point
- สร้างข้อความค้นหา 1 ย่อหน้าสั้น ๆ
- เน้น keyword สำคัญที่เกี่ยวกับหัวข้อ กลุ่มเป้าหมาย เป้าหมาย และปัญหา
- ใช้ถ้อยคำที่เหมาะกับการค้นหาหลักสูตรอบรม
- ไม่ต้องตอบยาว
- ไม่ต้อง bullet
- ไม่ต้องอธิบาย
    """.strip()

    user = f"""
topic: {topic}
target_group: {target_group}
goal: {goal}
pain_point: {pain_point}

ช่วยสร้าง query text สำหรับค้นหาหลักสูตรที่เกี่ยวข้อง
    """.strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )

        content = (response.choices[0].message.content or "").strip()

        return {
            "ok": True,
            "query_text": content if content else f"{topic} {target_group} {goal} {pain_point}",
        }

    except Exception:
        return {
            "ok": True,
            "query_text": f"{topic} {target_group} {goal} {pain_point}",
        }


async def get_embedding_python(text: str) -> dict:
    try:
        result = await call_openai_embedding_full(
            model="text-embedding-3-large",
            input_text=text,
        )

        return {
            "ok": True,
            "embedding": result["embedding"],
        }

    except Exception as e:
        print("DEBUG ERROR (embedding):", repr(e), flush=True)
        return {
            "ok": False,
            "error": f"Embedding API error: {str(e)}",
            "raw": "",
        }


def search_vector_db_python(query_vector: list[float]) -> dict:
    try:
        results = qdrant.query_points(
            collection_name="self_learning",
            query=query_vector,
            limit=5,
            with_payload=True,
        )

        print("[QDRANT RAW TYPE] =", type(results), flush=True)
        print("[QDRANT RAW] =", results, flush=True)

        points = getattr(results, "points", results)

        formatted = []
        for r in points:
            if isinstance(r, tuple):
                r = r[1] if len(r) > 1 else r[0]

            point_id = getattr(r, "id", None)
            score = getattr(r, "score", None)
            payload = getattr(r, "payload", {}) or {}

            formatted.append({
                "id": point_id,
                "score": score,
                "payload": payload,
            })

        print("[QDRANT RESULT COUNT] =", len(formatted), flush=True)

        return {
            "ok": True,
            "results": formatted,
        }

    except Exception as e:
        print("[QDRANT ERROR] =", repr(e), flush=True)
        return {
            "ok": False,
            "error": f"Vector search failed: {str(e)}",
        }


def get_scripts_by_script_ids(conn, script_ids) -> dict:
    out = {}
    cur = None

    if not isinstance(script_ids, list):
        script_ids = [script_ids]

    clean_ids = []
    seen = set()

    for sid in script_ids:
        sid = str(sid).strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        clean_ids.append(sid)

    if not clean_ids:
        return out

    placeholders = ",".join(["?"] * len(clean_ids))

    sql = f"""
        SELECT
            script_id,
            vdo_name,
            course_name,
            youtubelink,
            script
        FROM ai_idp_script
        WHERE script_id IN ({placeholders})
    """

    try:
        cur = conn.cursor()
        cur.execute(sql, clean_ids)

        columns = [col[0] for col in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    except Exception as e:
        print(f"[DB ERROR] get_scripts_by_script_ids: {e}", flush=True)
        return out

    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass

    for row in rows:
        sid = str(row.get("script_id", "")).strip()
        if sid:
            out[sid] = row

    return out


def attach_script_to_results(results: list[dict], script_map: dict) -> list[dict]:
    out = []

    for r in results:
        sid = str(r.get("payload", {}).get("script_id", "")).strip()

        if sid and sid in script_map:
            s = script_map[sid]
            r["script"] = s.get("script", "")
            r["vdo_name"] = s.get("vdo_name", "")
            r["course_name"] = s.get("course_name", "")
            r["youtubelink"] = s.get("youtubelink", "")
        else:
            r["script"] = ""
            r["vdo_name"] = ""
            r["course_name"] = ""
            r["youtubelink"] = ""

        out.append(r)

    return out


def build_context_from_vector_results(results: list[dict], limit: int = 3) -> str:
    context_blocks = []
    i = 0

    for row in results:
        if i >= limit:
            break

        name = str(row.get("vdo_name", "")).strip()
        course_name = str(row.get("course_name", "")).strip()
        script = str(row.get("script", "")).strip()
        score = row.get("score", "")
        retrival_text = str(row.get("payload", {}).get("retrival_text", "")).strip()

        if name == "" and course_name == "" and script == "":
            continue

        block = (
            f"ลำดับที่ {i + 1}\n"
            f"ชื่อเรื่อง: {name}\n"
            f"ชื่อหลักสูตร: {course_name}\n"
            f"ข้อความค้นคืน: {retrival_text}\n"
            f"คะแนนความใกล้เคียง: {score}\n"
            f"เนื้อหา: {script}"
        )

        context_blocks.append(block)
        i += 1

    return "\n\n-------------------------\n\n".join(context_blocks)


def build_followup_topics(results: list[dict], limit: int = 2) -> list[str]:
    topics = []
    i = 0

    for row in results:
        if i >= limit:
            break

        name = str(row.get("vdo_name", "")).strip()
        if name == "":
            continue

        topics.append(name)
        i += 1

    return topics


def generate_final_sale_reply(
    user_message: str,
    topic: str,
    target_group: str,
    goal: str,
    pain_point: str,
    query_text: str,
    context: str,
    follow,
) -> dict:
    system = """คุณคือ AI Sale Assistant ของสถาบันฝึกอบรม

กติกา:
- ใช้ "Context" เท่านั้นในการตอบเนื้อหาหลัก
- ห้ามแต่งข้อมูลหลักสูตรเกินจาก Context
- ตอบในเชิง "คัดและแนะนำหลักสูตร" ให้เหมาะกับความต้องการของผู้ใช้
- ต้องเชื่อมกับ topic, target_group, goal, pain_point
- หาก Context มีหลายรายการ ให้เลือกเฉพาะรายการที่เกี่ยวข้องที่สุด
- อธิบายให้กระชับ เข้าใจง่าย และดูเป็นมืออาชีพ
- ถ้าข้อมูลยังไม่พอ ให้บอกตรง ๆ ว่าพบข้อมูลใกล้เคียง แต่ยังไม่เพียงพอ
- ใช้โทนเหมือนเจ้าหน้าที่แนะนำหลักสูตรที่เข้าใจความต้องการลูกค้า
- จัด format เป็น HTML แบบเรียบง่าย เช่น div, p, ul, li, strong

โครงสร้างคำตอบ:
1. เริ่มจากสรุปความต้องการของผู้ใช้แบบสั้น ๆ
2. แนะนำหลักสูตรหรือเนื้อหาที่เกี่ยวข้องที่สุดจาก Context
3. อธิบายว่าทำไมถึงเหมาะกับกลุ่มเป้าหมายและปัญหานี้
4. ตอนท้ายชวนคุยต่อหรือชวนให้คัดเหลือ 1-2 หลักสูตร

ข้อห้าม:
- ห้ามตอบเป็นรายงาน
- ห้ามใช้ข้อมูลนอก Context
- ห้ามใช้หัวข้อที่เกี่ยวข้องเพิ่มเติมมาเป็นสาระหลัก
""".strip()

    user = f"""ข้อความล่าสุดของผู้ใช้: {user_message}

topic: {topic}
target_group: {target_group}
goal: {goal}
pain_point: {pain_point}
query_text: {query_text}

Context:
{context}

หัวข้อที่เกี่ยวข้องเพิ่มเติม:
{follow}

ช่วยตอบในรูปแบบแนะนำหลักสูตรอย่างเป็นธรรมชาติ กระชับ และเหมาะกับงานขาย
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )

        content = (response.choices[0].message.content or "").strip()

        return {
            "ok": True,
            "content": content,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


async def answer_when_sale_data_complete(
    conn,
    user_message: str,
    topic: str,
    target_group: str,
    goal: str,
    pain_point: str,
) -> dict:
    # 1) สร้าง query text สำหรับค้นหาหลักสูตร
    q = build_sale_query_text_by_ai(
        topic=topic,
        target_group=target_group,
        goal=goal,
        pain_point=pain_point,
    )
    if not q["ok"]:
        return {
            "ok": False,
            "reply": "เข้าใจความต้องการเบื้องต้นแล้ว แต่ยังสร้างข้อความค้นหาไม่ได้",
        }
    query_text = q["query_text"]

    # 2) embedding
    emb = await get_embedding_python(query_text)
    if not emb["ok"]:
        return {
            "ok": False,
            "reply": "เข้าใจความต้องการแล้ว แต่ยังสร้าง embedding ไม่สำเร็จ",
        }
    query_vector = emb["embedding"]

    # 3) search vector db
    search = search_vector_db_python(query_vector)
    print("[AI SALE SEARCH DEBUG] =", search, flush=True)

    if (not search["ok"]) or (not search["results"]):
        return {
            "ok": False,
            "reply": (
                "ตอนนี้ผมเข้าใจความต้องการของคุณแล้ว แต่ยังไม่พบข้อมูลหลักสูตรที่ใกล้เคียงเพียงพอ "
                "หากต้องการ คุณสามารถเล่ารายละเอียดเพิ่มอีกนิด เพื่อให้ผมช่วยคัดให้แคบลงได้ครับ"
            ),
        }

    # 4) ดึงข้อมูลเต็มจาก DB ด้วย script_id
    ids = []
    for r in search["results"]:
        script_id = r.get("payload", {}).get("script_id")
        if script_id:
            ids.append(script_id)

    script_map = get_scripts_by_script_ids(conn, ids)
    results_full = attach_script_to_results(search["results"], script_map)

    top_answer = results_full[:3]
    top_follow = results_full[3:5]

    # 5) build context
    context = build_context_from_vector_results(top_answer, 3)
    follow = build_followup_topics(top_follow, 2)

    # 6) generate final sale reply
    final = generate_final_sale_reply(
        user_message=user_message,
        topic=topic,
        target_group=target_group,
        goal=goal,
        pain_point=pain_point,
        query_text=query_text,
        context=context,
        follow=follow,
    )

    if not final["ok"]:
        return {
            "ok": False,
            "reply": "พบข้อมูลที่เกี่ยวข้องแล้ว แต่ยังไม่สามารถสรุปคำแนะนำหลักสูตรได้ในขณะนี้",
        }

    return {
        "ok": True,
        "reply": final["content"].strip(),
        "query_text": query_text,
        "context": context,
        "results": search["results"],
    }