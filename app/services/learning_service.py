from dotenv import load_dotenv
load_dotenv()

import json
import re
import requests
from openai import OpenAI
from qdrant_client import QdrantClient

qdrant = QdrantClient(
    url="https://f9a8611d-3692-4b14-bd09-bfaa135fe05d.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTllMzhiNzAtNTgyOS00MDUyLTgxNzMtNzMwZTIzMjgxZTA2In0.ktGpvSyHUg_44H6ihEdwthPfY6OG_0qU3zxb4hT5FNQ",
)

client = OpenAI()


from app.schemas import ChatState, LearningProgress
from app.services.ai_service import call_openai_chat


async def analyze_learning_progress(user_message: str, state: ChatState, model: str = "gpt-4.1-mini"):
    current_topic = str(state.topic or "unknown").strip()
    current_learning_need = str(state.learning_need or "unknown").strip()
    current_competency = str(state.learning_need or "unknown").strip()
    current_consulting_type = str(state.consulting_type or "unknown").strip()
    last_question = str(state.last_question or "none").strip()

    system_prompt = f"""        คุณคือ AI ที่สรุปข้อมูลการเรียนรู้จากข้อความผู้ใช้

        =====================
        บริบท
        =====================
        - topic ปัจจุบัน: {current_topic}
        - competency ปัจจุบัน: {current_competency}
        - consulting_type ปัจจุบัน: {current_consulting_type}
        - learning_need ปัจจุบัน: {current_learning_need}
        - last_question: {last_question}

        =====================
        หน้าที่
        =====================
        วิเคราะห์ "ข้อความล่าสุดของผู้ใช้" แล้วสรุป:

        1. topic = หัวข้อที่ผู้ใช้พูดถึงอย่างชัดเจน
        2. competency = ทักษะหลักที่เกี่ยวข้องกับ topic
        3. learning_need = สิ่งที่ผู้ใช้อยากรู้เกี่ยวกับ topic หรือสิ่งที่อยากพัฒนา

        =====================
        ค่าที่อนุญาตสำหรับ learning_need
        =====================
        - คืออะไร
        - อะไรบ้าง
        - เป็นอย่างไร
        - ต่างกันอย่างไร
        - มีอะไรบ้าง
        - ต้องทำอย่างไร
        - unknown

        =====================
        ค่าที่อนุญาตสำหรับ competency
        =====================
        - Presentation Skill
        - Communication Skill
        - Time Management
        - Self Confidence
        - Analytical Thinking
        - Strategic Thinking
        - Teamwork
        - Leadership
        - Problem Solving
        - Decision Making
        - unknown
        =====================
        กฎการวิเคราะห์ competency
        =====================
        1. competency คือ "ทักษะหลัก" ที่เกี่ยวข้องกับ topic
        - ไม่ใช่การคัดลอก topic มาตอบตรง ๆ
        - ไม่ใช่ประโยคยาว
        - ให้เลือกจากค่าที่อนุญาตเท่านั้น

        2. หากยังไม่สามารถระบุทักษะหลักได้ชัดเจน → ให้ตอบ "unknown"

        3. แนวทางการจับ competency
        - ถ้าเกี่ยวกับการพูดต่อหน้าคน, การพูดในที่ประชุม, การนำเสนอ
        → Presentation Skill
        - ถ้าเกี่ยวกับการสื่อสาร, การอธิบาย, การคุยให้คนเข้าใจ, การสื่อสารกับผู้อื่น
        → Communication Skill
        - ถ้าเกี่ยวกับงานล้น, การจัดลำดับงาน, การคุมเวลา, การบริหารเวลา
        → Time Management
        - ถ้าเกี่ยวกับความกลัว, ไม่มั่นใจ, ไม่กล้าแสดงออก
        → Self Confidence
        - ถ้าเกี่ยวกับการคิดวิเคราะห์, การแยกแยะข้อมูล, การหาสาเหตุ
        → Analytical Thinking
        - ถ้าเกี่ยวกับการคิดระยะยาว, การมองภาพรวม, การวางแผนเชิงอนาคต
        → Strategic Thinking
        - ถ้าเกี่ยวกับการทำงานร่วมกัน, การประสานงาน, การทำงานเป็นทีม
        → Teamwork
        - ถ้าเกี่ยวกับการนำทีม, การดูแลทีม, การบริหารคน
        → Leadership
        - ถ้าเกี่ยวกับการแก้ปัญหา, การรับมือสถานการณ์, การหาทางออก
        → Problem Solving
        - ถ้าเกี่ยวกับการตัดสินใจ, การเลือกแนวทาง, การชั่งน้ำหนักทางเลือก
        → Decision Making

        4. ถ้า topic เป็นเรื่องกว้างมาก หรือยังไม่ชัด
        → competency = "unknown"

        =====================
        ค่าที่อนุญาตสำหรับ consulting_type
        =====================
        - problem
        - emotion
        - skill
        - situation
        - case
        - unknown
        =====================
        กฎการวิเคราะห์ consulting_type
        =====================

        - ถ้า user แสดงความรู้สึก เช่น ไม่มั่นใจ เครียด กังวล กลัว
        → emotion

        - ถ้า user มีปัญหา เช่น ทำไม่ได้ งานล้น แก้ไม่ได้
        → problem

        - ถ้า user บอกว่าอยากพัฒนา อยากเก่งขึ้น
        → skill

        - ถ้า user ถามวิธีรับมือสถานการณ์
        → situation

        - ถ้า user ขอแนวทาง ตัวอย่าง
        → case

        - ถ้าไม่แน่ใจ → unknown

        =====================
        กฎการเก็บเป้าหมาย (Outcome) เพิ่มเติม
        =====================
        - หากผู้ใช้มีเป้าหมาย เช่น "เพื่อ...", "ให้...", "จะได้..."
        → ต้องรวมเป้าหมายเข้าไปใน learning_need

        - ห้ามตัดคำสำคัญ เช่น:
        "ทำงานเสร็จเร็วขึ้น", "พูดให้มั่นใจขึ้น", "ขายได้มากขึ้น"

        - รูปแบบที่ถูกต้อง:
        "ต้องทำอย่างไรเพื่อให้..."

        =====================
        การตีความคำตอบสั้น (สำคัญมาก)
        =====================
        - ถ้าข้อความของผู้ใช้เป็นคำตอบสั้น ๆ เช่น:
        "ใช่", "โอเค", "อยากรู้", "เอา", "ได้", "อันนี้"
        → ให้ใช้ last_question เป็นบริบทในการตีความ

        - ถ้า last_question เป็นคำถามเกี่ยวกับ "วิธี", "การทำ", "การแก้ปัญหา"
        → ให้ learning_need = "ต้องทำอย่างไร"

        - ห้ามตอบว่า unknown ในกรณีนี้

        
        

        =====================
        กฎสำคัญ (ต้องทำตามอย่างเคร่งครัด)
        =====================

        1. ห้ามเดา
        - ห้ามเดาข้อมูลใหม่ที่ผู้ใช้ไม่ได้พูด
        - แต่สามารถเรียบเรียงหรือรวมประโยคเพื่อให้ได้ความหมายที่สมบูรณ์
        - ถ้าผู้ใช้ไม่ได้ถามหรือไม่ได้บอกว่าต้องการรู้อะไร → ต้องตอบ learning_need = "unknown"

        2. แยก "เล่าปัญหา" ออกจาก "ถามคำถาม"
        - ถ้าเป็นการเล่าปัญหา เช่น:
        "มีปัญหา...", "กังวล...", "ไม่มั่นใจ...", "รู้สึกว่า..."
        → ถือว่ายังไม่มี learning_need → ต้องเป็น "unknown"

        3. learning_need จะมีได้เฉพาะกรณีนี้เท่านั้น:
        - ผู้ใช้ถามคำถามโดยตรง เช่น "คืออะไร", "ทำอย่างไร", "มีอะไรบ้าง"
        - หรือระบุชัดว่าอยากรู้/อยากพัฒนา เช่น "อยากรู้วิธี...", "ควรทำอย่างไร" , "อยากพัฒนา"

        4. ถ้าไม่แน่ใจ → ให้ตอบ "unknown" เท่านั้น (ห้ามเลือกค่าอื่น)

        5. topic:
        - ถ้ามีหัวข้อชัด → ให้ระบุ
        - ถ้ายังไม่ชัด → "unknown"
        - ถ้าเป็นหัวข้อกว้างๆอย่างอยากพัฒนาตัวเอง → "unknown"

        =====================
        ตัวอย่าง
        =====================

        ผู้ใช้: "กังวลเรื่องงาน"
        {{
        "topic": "ความกังวลเรื่องงาน",
        "competency": "unknown",
        "learning_need": "unknown"
        }}

        ผู้ใช้: "ไม่มั่นใจเวลาพูดในที่ประชุม"
        {{
        "topic": "ความไม่มั่นใจในการพูดในที่ประชุม",
        "competency": "Presentation Skill",
        "learning_need": "unknown"
        }}

        ผู้ใช้: "การสื่อสารที่ดีคืออะไร"
        {{
        "topic": "การสื่อสาร",
        "competency": "Communication Skill",
        "learning_need": "คืออะไร"
        }}

        ผู้ใช้: "อยากพัฒนาการพูด ควรทำอย่างไร"
        {{
        "topic": "การพัฒนาการพูด",
        "competency": "Presentation Skill",
        "learning_need": "ต้องทำอย่างไร"
        }}

        ผู้ใช้: "สวัสดีฉันอยากพัฒนาตนเอง"
        {{
        "topic": "unknown",
        "competency": "unknown",
        "learning_need": "ต้องทำอย่างไร"
        }}

        ผู้ใช้: "อยากจัดการเวลาให้ดีขึ้น"
        {{
        "topic": "การจัดการเวลา",
        "competency": "Time Management",
        "learning_need": "ต้องทำอย่างไรเพื่อให้จัดการเวลาได้ดีขึ้น"
        }}

        =====================
        รูปแบบคำตอบ
        =====================
        ตอบเป็น JSON เท่านั้น:

        {{
        "topic": "string",
        "competency": "string",
        "consulting_type": "string",
        "learning_need": "string"
        }}"""

    user_prompt = "ข้อความผู้ใช้: " + user_message

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    text = response.choices[0].message.content or ""
    text = re.sub(r"```json|```", "", text)
    decoded = json.loads(text.strip()) if text.strip() else {}

    topic = decoded.get("topic", "unknown")
    learning_need = decoded.get("learning_need", "unknown")
    competency = decoded.get("competency", "unknown")
    consulting_type = decoded.get("consulting_type", "unknown")

    # merge state (สำคัญ!)
    if topic == "unknown":
        topic = current_topic

    if competency == "unknown":
        competency = current_competency

    if learning_need == "unknown":
        learning_need = current_learning_need

    if consulting_type == "unknown":
        consulting_type = current_consulting_type

    # derive next_action
    if topic == "unknown":
        next_action = "ask_topic"
    elif learning_need == "unknown":
        next_action = "ask_learning_need"
    else:
        next_action = "ready"

    return LearningProgress(
    topic=topic,
    competency=competency,
    consulting_type=consulting_type,
    learning_need=learning_need,
    last_question=last_question,
    next_action=next_action,
    raw=text,
)


async def generate_learning_question(
    user_message: str,
    state: ChatState,
    question_type: int,
    model: str = "gpt-4.1-mini",
) -> str:
    """
    question_type:
    1 = ask topic
    2 = ask learning need
    3 = ask topic when learning_need exists
    """

    topic = (state.topic or "unknown").strip()
    last_question = (state.last_question or "unknown").strip()
    learning_need = (state.learning_need or "unknown").strip()

    if question_type == 1:
        missing_info = "ยังไม่รู้ว่าผู้ใช้ต้องการเรียนรู้หรือปรึกษาเรื่องอะไร"
        goal = (
            "ถามต่ออย่างเป็นธรรมชาติเพื่อให้ผู้ใช้ระบุหัวข้อที่อยากเรียนรู้หรืออยากปรึกษา "
            "และ learning need ให้ชัดขึ้นในด้านดังนี้ "
            "คืออะไร|อะไรบ้าง|เป็นอย่างไร|ต่างกันอย่างไร|มีอะไรบ้าง|ต้องทำอย่างไร| "
            "โดยเลือกใช้ให้เหมาะสมกบบริบทของหัวข้อ"
        )
    elif question_type == 2:
        missing_info = "รู้หัวข้อแล้ว แต่ยังไม่รู้ว่าผู้ใช้อยากเรียนรู้มุมไหนหรืออยากได้ความช่วยเหลือด้านไหน"
        goal = (
            "ให้ AI เลือก learning_need ที่เหมาะที่สุด จาก "
            "คืออะไร|อะไรบ้าง|เป็นอย่างไร|ต่างกันอย่างไร|มีอะไรบ้าง|ต้องทำอย่างไร| "
            "แล้วถามนำไปทางนั้นโดยเลือกใช้ให้เหมาะสมกบบริบทของหัวข้อ"
        )
    elif question_type == 3:
        missing_info = "รู้สิ่งที่ผู้ใช้อยากเรียนรู้แล้ว แต่ยังไม่รู้ว่ากำลังพูดถึงหัวข้ออะไร"
        goal = "ถามต่ออย่างเป็นธรรมชาติเพื่อให้ผู้ใช้ระบุหัวข้อให้ชัด"
    else:
        missing_info = "ข้อมูลครบแล้ว"
        goal = "ไม่ต้องถามเพิ่ม"

    system_prompt = f"""คุณคือ AI Coaching Assistant ที่กำลังคุยกับผู้ใช้ในบทสนทนาเดียวกันอย่างต่อเนื่อง

หน้าที่:
- อ่านข้อความล่าสุดของผู้ใช้
- ดูว่าตอนนี้รู้อะไรแล้ว และยังขาดอะไร
- สร้าง “คำถามถัดไป” ที่ต่อเนื่องจากสิ่งที่ผู้ใช้เพิ่งพูด
- ต้องทำให้การสนทนาดูเป็นธรรมชาติ เหมือนกำลังคุยเรื่องเดียวกันอยู่จริง
- ถ้าผู้ใช้พูดมาบางส่วนแล้ว ให้เชื่อมจากสิ่งที่ผู้ใช้พูดก่อน แล้วค่อยถามสิ่งที่ยังขาด

ข้อมูลปัจจุบัน:
- topic: {topic}
- learning_need: {learning_need}
- สิ่งที่ยังขาด: {missing_info}
- เป้าหมายของคำถามนี้: {goal}

=====================
หลักการเลือก learning_need
=====================
- ให้ AI วิเคราะห์ว่าจากข้อความผู้ใช้ learning_need แบบใด “เหมาะสมที่สุด”
- ให้เลือกเพียง 1 แบบเท่านั้น
- ห้ามเสนอหลายตัวเลือก
- ห้ามถามแบบ A หรือ B

=====================
วิธีถาม
=====================
- เมื่อเลือก learning_need แล้ว ให้สร้างคำถามที่ "พาไปสู่มุมนั้น"
- ห้ามบอกชื่อ learning_need ตรง ๆ
- ห้ามเสนอหลายทางเลือก
- ต้องถามแบบเนียน ๆ เหมือนโค้ช

ตัวอย่าง:

ผู้ใช้: "เวลาไม่พอทำงานไม่ทัน"
→ learning_need ที่เหมาะ: "ต้องทำอย่างไร"

คำถามที่ดี:
"จากที่คุณเล่ามา คุณอยากเรียนรู้วิธีการบริการจัดการเวลาไหมครับ?"

คำถามที่ห้าม:
"คุณอยากวางแผนเวลา หรือจัดการสิ่งรบกวนดี?"

หลักการตอบ:
- ใช้ข้อมูลจากข้อความล่าสุดของผู้ใช้มาประกอบคำถาม
- ถ้าควรอ้างอิงสิ่งที่ผู้ใช้พูด ให้พูดสั้น ๆ เช่น "จากที่คุณเล่ามา..." หรือ "ถ้าเป็นเรื่องนี้..."
- ถามเฉพาะสิ่งที่ยังขาดอยู่เท่านั้น
- ให้คำถามช่วยให้ผู้ใช้ตอบง่ายขึ้น
- ต้องฟังดูเป็นคนคุยต่อ ไม่ใช่ระบบถามเป็นข้อ ๆ
- ถ้าเหมาะสม สามารถยกตัวอย่างสั้น ๆ ได้ แต่ต้องไม่ยาว

ข้อห้าม:
- ห้ามตอบยาว
- ห้ามถามหลายคำถามซ้อน
- ห้ามแข็งหรือเป็นทางการเกินไป
- ห้ามถามแบบหุ่นยนต์
- ห้ามสรุปเกินจากที่ผู้ใช้พูด
- ถ้าข้อมูลครบแล้ว ห้ามถามเพิ่มโดยไม่จำเป็น

รูปแบบคำตอบ:
- ตอบเป็นภาษาไทย
- 1 ประโยค หรือไม่เกิน 2 ประโยคสั้น
- ต้องเป็นคำถามที่ชวนให้ผู้ใช้ตอบต่อได้ง่าย"""

    user_prompt = f"""คำถามจาก โค้ชล่าสุด: {last_question}
ข้อความล่าสุดของผู้ใช้: {user_message}

ช่วยสร้างคำถามถัดไปที่ต่อเนื่องจากข้อความนี้ โดยถามเฉพาะข้อมูลที่ยังขาดอยู่"""

    content = await call_openai_chat(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
    )

    return (content or "ช่วยเล่าเพิ่มเติมอีกนิดได้ไหมครับ").strip()

def build_query_text_by_ai(topic: str, competency: str, learning_need: str) -> dict:
    system = """คุณคือ AI ที่ช่วยสร้าง query text สำหรับใช้ค้นหาข้อมูลใน vector database

    กติกา:
    - รับ topic, competency และ learning_need
    - สร้างข้อความค้นหา 1 ย่อหน้าสั้น ๆ
    - เน้น keyword สำคัญที่เกี่ยวกับหัวข้อและความต้องการเรียนรู้
    - ไม่ต้องตอบยาว
    - ไม่ต้องใส่ bullet
    - ไม่ต้องอธิบายเกินความจำเป็น"""

    user = (
        f"topic: {topic}\n"
        f"competency: {competency}\n"
        f"learning_need: {learning_need}\n\n"
        "ช่วยสร้าง query text สำหรับค้นหา knowledge ที่เกี่ยวข้อง"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",   # หรือใช้ MODEL_NAME_ROUTER
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        return {
            "ok": True,
            "query_text": content,
        }

    except Exception as e:
        # ⚠️ ต้อง fallback เหมือน PHP
        return {
            "ok": True,
            "query_text": f"หัวข้อ {topic} ความต้องการเรียนรู้ {learning_need}",
        }

def get_embedding_python(text: str) -> dict:
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )

        embedding = response.data[0].embedding

        return {
            "ok": True,
            "embedding": embedding,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"Embedding API error: {str(e)}",
            "raw": ""
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
            payload = getattr(r, "payload", {})

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
            "error": f"Vector search failed: {str(e)}"
        }

def get_scripts_by_script_ids(conn, script_ids) -> dict:
    out = {}
    cur = None

    # normalize → list
    if not isinstance(script_ids, list):
        script_ids = [script_ids]

    # clean + unique
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

    placeholders = ",".join(["?"] * len(clean_ids))   # ถ้าใช้ sqlite3
    # placeholders = ",".join(["%s"] * len(clean_ids))  # ถ้าใช้ mysql.connector

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
            cur.close()

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
        script = str(row.get("script", "")).strip()
        score = row.get("score", "")
        retrival_text = str(row.get("payload", {}).get("retrival_text", "")).strip()

        if name == "" and script == "":
            continue

        block = (
            f"ลำดับที่ {i + 1}\n"
            f"ชื่อเรื่อง: {name}\n"
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

def generate_final_learning_reply(
    user_message: str,
    topic: str,
    competency: str,
    learning_need: str,
    query_text: str,
    context: str,
    follow
) -> dict:

    system = """คุณคือ AI Coach ที่ช่วยตอบจากข้อมูล knowledge ที่มี

กติกา:
- ใช้ "Context" เท่านั้นในการตอบคำถามหลัก
- ห้ามนำ "Topic ที่เกี่ยวข้อง" มาใช้เป็นสาระหลักของคำตอบ
- "Topic ที่เกี่ยวข้อง" ใช้ได้เฉพาะช่วงท้าย เพื่อชวนผู้ใช้ต่อยอดบทสนทนา
- ตอบให้สัมพันธ์กับ topic และ learning_need
- หาก Context มีหลายเรื่อง ให้สรุปเฉพาะส่วนที่เกี่ยวข้องมากที่สุด
- ตอบให้เข้าใจง่าย กระชับ และนำไปใช้ได้
- หากข้อมูลยังไม่พอ ให้บอกอย่างตรงไปตรงมาว่าพบข้อมูลใกล้เคียง แต่ยังไม่เพียงพอ
- ห้ามแต่งรายละเอียดเกินจาก Context
- จัด format สำหรับแสดงใน HTML โดยใช้แท็กที่เรียบง่าย เช่น div, p, ul, li, strong
- เน้นคำสำคัญและหัวข้อให้อ่านง่าย
- ตอนท้ายค่อยชวนผู้ใช้ต่อยอดจาก Topic ที่เกี่ยวข้องอย่างเป็นธรรมชาติ"""

    user = f"""คำถามผู้ใช้: {user_message}

Topic: {topic}
competency: {competency}
Learning Need: {learning_need}
Query Text ที่ใช้ค้นหา: {query_text}

Context สำหรับใช้ตอบคำถามหลัก:
{context}

Topic ที่เกี่ยวข้อง (ใช้เฉพาะสำหรับชวนคุยต่อท้ายคำตอบ ไม่ใช้เป็นข้อมูลหลัก):
{follow}

ช่วยตอบผู้ใช้โดยให้อธิบายว่าสิ่งที่ผู้ใช้กำลังติดอยู่คือ competency ด้านใด ลักษะไหนที่ผู้ใช้ติดอยู่ แล้วค่อยแนะนำวิธีแก้ปัญหาโดยสรุปสาระสำคัญจาก Context ให้เป็นธรรมชาติไม่ให้เหมือนโปรแกรมให้เหมือนโค้ชคุยมากกว่า"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        return {
            "ok": True,
            "content": content,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }
    
def answer_when_learning_data_complete(
    conn,
    user_message: str,
    topic: str,
    competency: str,
    learning_need: str,
) -> dict:
    
    # 1) ให้ AI สร้าง query text
    q = build_query_text_by_ai(topic, competency, learning_need)
    if not q["ok"]:
        return {
            "ok": False,
            "reply": "พบหัวข้อและความต้องการแล้ว แต่ยังสร้างข้อความค้นหาไม่ได้",
        }
    query_text = q["query_text"]

    # 2) embedding
    emb = get_embedding_python(query_text)
    if not emb["ok"]:
        return {
            "ok": False,
            "reply": "พบหัวข้อและความต้องการแล้ว แต่ยังสร้าง embedding ไม่สำเร็จ",
        }
    query_vector = emb["embedding"]

    # 3) search vector db
    search = search_vector_db_python(query_vector)
    print("[SEARCH DEBUG] =", search, flush=True)

    if (not search["ok"]) or (not search["results"]):
        return {
            "ok": False,
            "reply": "พบหัวข้อและความต้องการแล้ว แต่ยังไม่พบข้อมูลที่เกี่ยวข้องเพียงพอ",
        }

    ids = []

    for r in search["results"]:
        script_id = r.get("payload", {}).get("script_id")
        if script_id:
            ids.append(script_id)

    script_map = get_scripts_by_script_ids(conn, ids)

    results_full = attach_script_to_results(search["results"], script_map)

    top_answer = results_full[:3]
    top_follow = results_full[3:5]

    # 4) build context
    context = build_context_from_vector_results(top_answer, 3)
    follow = build_followup_topics(top_follow, 2)

    # 5) generate final reply
    final = generate_final_learning_reply(
        user_message=user_message,
        topic=topic,
        competency=competency,
        learning_need=learning_need,
        query_text=query_text,
        context=context,
        follow=follow,
    )

    if not final["ok"]:
        return {
            "ok": False,
            "reply": "พบข้อมูลที่เกี่ยวข้องแล้ว แต่ยังไม่สามารถสรุปคำตอบได้ในขณะนี้",
        }

    return {
        "ok": True,
        "reply": final["content"].strip(),
        "query_text": query_text,
        "context": context,
        "results": search["results"],
    }