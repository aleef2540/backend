from dotenv import load_dotenv
load_dotenv()

import json
import re
import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from typing import Any

qdrant = QdrantClient(
    url="https://f9a8611d-3692-4b14-bd09-bfaa135fe05d.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTllMzhiNzAtNTgyOS00MDUyLTgxNzMtNzMwZTIzMjgxZTA2In0.ktGpvSyHUg_44H6ihEdwthPfY6OG_0qU3zxb4hT5FNQ",
)

client = OpenAI()


from app.schemas_model1 import ChatState, LearningProgress
# from app.services.ai_service import call_openai_chat_full
from app.services.call_ai import call_openai_chat_full, call_openai_embedding_full


async def analyze_learning_progress(
    user_message: str,
    state: ChatState,
    model: str = "gpt-4.1-mini"
    ) -> LearningProgress:
    current_topic = str(getattr(state, "topic", "unknown") or "unknown").strip()
    current_goal = str(getattr(state, "goal", "unknown") or "unknown").strip()
    current_event = str(getattr(state, "event", "unknown") or "unknown").strip()
    current_learning_need = str(getattr(state, "learning_need", "unknown") or "unknown").strip()
    last_question = str(getattr(state, "last_question", "none") or "none").strip()

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
        if not text or not text.strip():
            return {}

        cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

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

    system_prompt = f"""
        คุณคือ AI ที่ทำหน้าที่สกัดข้อมูลการเรียนรู้จาก "ข้อความล่าสุดของผู้ใช้"
        ให้แยกข้อมูลออกเป็น 4 ช่องคือ topic, goal, event, learning_need

        =====================
        บริบทปัจจุบัน
        =====================
        - topic ปัจจุบัน: {current_topic}
        - goal ปัจจุบัน: {current_goal}
        - event ปัจจุบัน: {current_event}
        - learning_need ปัจจุบัน: {current_learning_need}
        - last_question: {last_question}

        =====================
        ความหมายของแต่ละ field
        =====================
        1. topic
        = หัวข้อหลักที่ผู้ใช้กำลังพูดถึง
        = ตอบคำถามว่า "ผู้ใช้กำลังพูดเรื่องอะไร"

        2. goal
        = สิ่งที่ผู้ใช้อยากทำให้ดีขึ้น / ผลลัพธ์ที่อยากได้
        = ตอบคำถามว่า "ผู้ใช้อยากไปถึงอะไร"

        3. event
        = สถานการณ์ / บริบท / เหตุการณ์ที่ผู้ใช้กำลังเจอ หรือจะต้องนำเรื่องนี้ไปใช้
        = ตอบคำถามว่า "เรื่องนี้เกิดขึ้นเมื่อไร / ที่ไหน / ในสถานการณ์ใด"

        4. learning_need
        = สิ่งที่ผู้ใช้อยากรู้ / อยากได้คำแนะนำ / อยากให้ช่วย ณ ตอนนี้
        = ตอบคำถามว่า "ตอนนี้ผู้ใช้อยากให้ช่วยเรื่องอะไร"

        =====================
        กฎการแยก field
        =====================
        - ข้อความเดียวอาจมีมากกว่า 1 field ได้
        - ให้ดึงทุก field ที่พบ
        - ถ้า field ไหนไม่ชัดเจนจริง ๆ ให้ตอบ "unknown"
        - ห้ามเดาข้อมูลใหม่ที่ผู้ใช้ไม่ได้สื่อ
        - แต่อนุญาตให้ "สรุปถ้อยคำ" จากสิ่งที่ผู้ใช้พูดได้ โดยห้ามเพิ่มความหมายใหม่เกินข้อความ

        =====================
        กฎสำคัญเรื่อง goal
        =====================
        - goal = ผลลัพธ์ที่ผู้ใช้อยากได้
        - ถ้าผู้ใช้เล่าปัญหาอย่างเดียว แต่ไม่ได้บอกผลลัพธ์ที่ต้องการชัดเจน
        ให้ตอบ goal = "unknown"

        ตัวอย่าง:
        - "อยากพูดในที่ประชุมได้ดีขึ้น"
        -> goal = "พูดในที่ประชุมได้ดีขึ้น"

        - "ไม่มั่นใจเวลา present งานต่อผู้บริหาร"
        -> goal = "unknown"

        =====================
        กฎสำคัญเรื่อง event
        =====================
        - event = สถานการณ์หรือบริบทที่เรื่องนี้เกิดขึ้น
        - มักมีคำบอกเวลา / สถานที่ / โอกาส / สถานการณ์ เช่น
        "พรุ่งนี้", "ตอนประชุม", "เวลา present", "กับลูกค้า", "ต่อหน้าผู้บริหาร"

        ตัวอย่าง:
        - "อยากตอบคำถามลูกค้าให้มั่นใจขึ้นตอนพรีเซนต์งาน"
        -> event = "ตอนพรีเซนต์งาน"

        - "พรุ่งนี้ต้อง present ต่อหน้าผู้บริหาร"
        -> event = "พรุ่งนี้ต้อง present ต่อหน้าผู้บริหาร"

        =====================
        กฎสำคัญเรื่อง learning_need
        =====================
        - learning_need จะต้องเป็นสิ่งที่ผู้ใช้อยากรู้หรืออยากให้ช่วยในเรื่องที่ "เฉพาะเจาะจงพอ"
        - ถ้าเป็นเพียงคำพูดกว้าง ๆ เช่น
        "มีเรื่องจะปรึกษา", "ขอคำแนะนำ", "ช่วยหน่อย", "มีคำถาม"
        แต่ยังไม่บอกว่าปรึกษาเรื่องอะไร
        ให้ตอบ learning_need = "unknown"
        - ถ้าไม่มีคำขอชัดเจน ให้ตอบ "unknown"

        ตัวอย่าง:
        - "อยากรู้ว่าจะเริ่มพูดยังไง"
        -> learning_need = "อยากรู้ว่าจะเริ่มพูดยังไง"

        - "ควรเริ่มพัฒนาจากตรงไหน"
        -> learning_need = "ควรเริ่มพัฒนาจากตรงไหน"

        - "มีเรื่องจะปรึกษา"
        -> learning_need = "unknown"

        =====================
        กฎสำคัญเรื่อง topic
        =====================
        - topic = ชื่อหัวข้อหลัก เช่น
        "การพูดในที่ประชุม", "การให้ feedback", "การบริหารทีม", "การสื่อสารกับลูกค้า"
        - ถ้าไม่ชัดจริง ๆ ให้ตอบ "unknown"

        =====================
        ตัวอย่าง
        =====================
        ข้อความ: "พรุ่งนี้ต้อง present งานต่อหน้าผู้บริหาร อยากพูดให้มั่นใจขึ้น แต่ยังไม่รู้จะเริ่มยังไง"
        ตอบ:
        {{
        "topic": "การ present งานต่อหน้าผู้บริหาร",
        "goal": "พูดให้มั่นใจขึ้น",
        "event": "พรุ่งนี้ต้อง present งานต่อหน้าผู้บริหาร",
        "learning_need": "ยังไม่รู้จะเริ่มยังไง"
        }}

        ข้อความ: "ช่วงนี้ทีมไม่ค่อยร่วมมือกัน ฉันอยากเป็นหัวหน้าที่บริหารทีมได้ดีขึ้น ควรเริ่มพัฒนาจากตรงไหน"
        ตอบ:
        {{
        "topic": "การบริหารทีม",
        "goal": "เป็นหัวหน้าที่บริหารทีมได้ดีขึ้น",
        "event": "ช่วงนี้ทีมไม่ค่อยร่วมมือกัน",
        "learning_need": "ควรเริ่มพัฒนาจากตรงไหน"
        }}

        ข้อความ: "ไม่มั่นใจเวลา present งานต่อผู้บริหาร"
        ตอบ:
        {{
        "topic": "การ present งานต่อผู้บริหาร",
        "goal": "unknown",
        "event": "เวลา present งานต่อผู้บริหาร",
        "learning_need": "unknown"
        }}

        =====================
        รูปแบบคำตอบ
        =====================
        ตอบเป็น JSON object เท่านั้น
        ห้ามมีคำอธิบายอื่น
        ใช้ format นี้เท่านั้น:

        {{
        "topic": "string",
        "goal": "string",
        "event": "string",
        "learning_need": "string"
        }}
        """.strip()

    user_prompt = f"ข้อความล่าสุดของผู้ใช้: {user_message}"

    try:
        # ถ้าใช้ async client ให้ใช้ await แบบนี้
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        text = response.choices[0].message.content or ""
    except Exception as e:
        raw_error = f"MODEL_CALL_ERROR: {str(e)}"
        return LearningProgress(
            topic=current_topic,
            goal=current_goal,
            event=current_event,
            learning_need=current_learning_need,
            last_question=last_question,
            next_action=(
                "ask_topic" if current_topic == "unknown"
                else "ask_goal" if current_goal == "unknown"
                else "ask_event" if current_event == "unknown"
                else "ask_learning_need" if current_learning_need == "unknown"
                else "ready"
            ),
            raw=raw_error,
        )

    decoded = extract_json_object(text)

    topic = normalize_text(decoded.get("topic", "unknown"))
    goal = normalize_text(decoded.get("goal", "unknown"))
    event = normalize_text(decoded.get("event", "unknown"))
    learning_need = normalize_text(decoded.get("learning_need", "unknown"))

    # merge state
    if topic == "unknown":
        topic = normalize_text(current_topic)

    if goal == "unknown":
        goal = normalize_text(current_goal)

    if event == "unknown":
        event = normalize_text(current_event)

    if learning_need == "unknown":
        learning_need = normalize_text(current_learning_need)

    # derive next_action
    if topic == "unknown":
        next_action = "ask_topic"
    elif goal == "unknown":
        next_action = "ask_goal"
    elif event == "unknown":
        next_action = "ask_event"
    elif learning_need == "unknown":
        next_action = "ask_learning_need"
    else:
        next_action = "ready"

    return LearningProgress(
        topic=topic,
        goal=goal,
        event=event,
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
    2 = ask goal
    3 = ask event
    4 = ask learning_need
    """

    topic = str(state.topic or "unknown").strip()
    goal_value = str(state.goal or "unknown").strip()
    event = str(state.event or "unknown").strip()
    learning_need = str(state.learning_need or "unknown").strip()
    last_question = str(state.last_question or "unknown").strip()

    if question_type == 1:
        missing_info = "ยังไม่รู้ว่าผู้ใช้อยากพัฒนาเรื่องอะไรเป็นหลัก"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุหัวข้อที่อยากพัฒนาให้ชัดเจนขึ้น "
            "เช่น การสื่อสาร การบริหารเวลา การคิดวิเคราะห์ ภาวะผู้นำ หรือหัวข้ออื่นที่เฉพาะขึ้น"
        )
        question_focus = "topic"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "อยากพูดในที่ประชุมได้ดีขึ้น"
        คำถามที่ดี:
        "เข้าใจว่าคุณอยากพูดได้ดีขึ้นในที่ประชุมครับ แล้วถ้ามองให้ชัดขึ้น ตอนนี้คุณอยากโฟกัสพัฒนาเรื่องอะไรเป็นหลักครับ"

        ผู้ใช้: "ทำงานช้าตลอด"
        คำถามที่ดี:
        "จากที่คุณเล่ามา ผมอยากเข้าใจให้ชัดขึ้นอีกนิดครับ ว่าตอนนี้คุณอยากพัฒนาเรื่องอะไรเป็นหลัก เช่น การบริหารเวลา การจัดลำดับงาน หรือเรื่องอื่นที่คุณรู้สึกติดอยู่ครับ"
        """

    elif question_type == 2:
        missing_info = "รู้หัวข้อแล้ว แต่ยังไม่รู้ว่าผู้ใช้อยากให้ตัวเองทำอะไรได้ดีขึ้น"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุผลลัพธ์ที่คาดหวังหรือสิ่งที่อยากทำได้ดีขึ้น "
            "โดยถามในมุมว่าอยากให้ตัวเองทำอะไรดีขึ้น ไม่ใช่ถามหัวข้อซ้ำ"
        )
        question_focus = "goal"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "อยากพัฒนาการสื่อสาร"
        คำถามที่ดี:
        "ถ้าเป็นเรื่องการสื่อสาร ผมอยากรู้เพิ่มอีกนิดครับว่าคุณอยากให้ตัวเองทำอะไรได้ดีขึ้นเป็นพิเศษ"

        ผู้ใช้: "อยากพัฒนาเรื่องการบริหารเวลา"
        คำถามที่ดี:
        "ถ้าเป็นเรื่องการบริหารเวลา ตอนนี้คุณอยากให้ตัวเองทำอะไรได้ดีขึ้นครับ เช่น ทำงานให้เสร็จเร็วขึ้น จัดลำดับงานได้ชัดขึ้น หรือคุมเวลาในแต่ละวันได้ดีขึ้น"
        """

    elif question_type == 3:
        missing_info = "รู้หัวข้อและเป้าหมายแล้ว แต่ยังไม่รู้ว่าจะนำไปใช้ในสถานการณ์ไหน"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุสถานการณ์ บริบท หรือเหตุการณ์ที่เจออยู่ "
            "เช่น ตอนประชุม ตอนคุยกับหัวหน้า ตอนพรีเซนต์งาน ตอนคุยกับลูกค้า หรือระหว่างทำงานจริง"
        )
        question_focus = "event"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "อยากสื่อสารให้ชัดขึ้น"
        คำถามที่ดี:
        "เรื่องนี้คุณอยากเอาไปใช้ในสถานการณ์ไหนเป็นหลักครับ เพื่อที่ผมจะได้ช่วยได้ตรงมากขึ้น"

        ผู้ใช้: "อยากจัดลำดับงานให้ดีขึ้น"
        คำถามที่ดี:
        "ปัญหานี้มักเกิดขึ้นในช่วงไหนของการทำงานครับ เช่น ตอนงานเข้าพร้อมกัน ตอนต้องเร่งส่งงาน หรือในสถานการณ์อื่นที่คุณเจอบ่อย ๆ"
        """

    elif question_type == 4:
        missing_info = "รู้เรื่องที่อยากพัฒนาแล้ว แต่ยังไม่ชัดว่าตอนนี้ผู้ใช้อยากให้ช่วยเรื่องอะไรโดยตรง"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้ระบุสิ่งที่อยากรู้ อยากได้คำแนะนำ "
            "หรืออยากให้ช่วย ณ ตอนนี้ เช่น อยากเริ่มยังไง ควรทำแบบไหนก่อน "
            "อยากได้แนวทาง วิธีคิด วิธีรับมือ หรือขั้นตอนที่นำไปใช้ได้จริง"
        )
        question_focus = "learning_need"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "อยากพัฒนาการสื่อสารกับทีม"
        คำถามที่ดี:
        "ได้เลยครับ ถ้าเป็นเรื่องการสื่อสารกับทีม ตอนนี้คุณอยากให้ผมช่วยในมุมไหนมากที่สุดครับ เช่น อยากรู้ว่าควรเริ่มปรับตรงไหนก่อน หรืออยากได้วิธีที่เอาไปใช้ได้เลย"

        ผู้ใช้: "พรุ่งนี้ต้อง present งานต่อผู้บริหาร"
        คำถามที่ดี:
        "เข้าใจเลยครับ ถ้าเป็นสถานการณ์นี้ ตอนนี้คุณอยากได้ความช่วยเหลือเรื่องไหนมากที่สุดครับ เช่น วิธีเริ่มพูด วิธีเรียบเรียงเนื้อหา หรือวิธีรับมือความตื่นเต้น"
        """

    else:
        return "ช่วยเล่าเพิ่มเติมอีกนิดได้ไหมครับ"

    system_prompt = f"""คุณคือ AI Coaching Assistant ที่กำลังคุยกับผู้ใช้ในบทสนทนาเดียวกันอย่างต่อเนื่อง

        หน้าที่:
        - อ่านข้อความล่าสุดของผู้ใช้
        - ดูว่าตอนนี้รู้อะไรแล้ว และยังขาดอะไร
        - สร้าง “คำถามถัดไป” ที่ต่อเนื่องจากสิ่งที่ผู้ใช้เพิ่งพูด
        - คำถามต้องดูเป็นธรรมชาติ เหมือนโค้ชกำลังคุยต่อเนื่อง
        - ถามเฉพาะข้อมูลที่ยังขาดอยู่เท่านั้น

        ข้อมูลปัจจุบัน:
        - topic: {topic}
        - goal: {goal_value}
        - event: {event}
        - learning_need: {learning_need}
        - สิ่งที่ยังขาด: {missing_info}
        - เป้าหมายของคำถามนี้: {goal_instruction}
        - focus ที่ต้องถาม: {question_focus}

        หลักการถาม:
        1. ถามเพียง 1 คำถามเท่านั้น
        2. ห้ามถามหลายประเด็นในประโยคเดียว
        3. ห้ามถามซ้ำสิ่งที่มีอยู่แล้ว
        4. ถ้าผู้ใช้พูดมาบางส่วนแล้ว ให้เชื่อมจากสิ่งที่ผู้ใช้พูดก่อน แล้วค่อยถามสิ่งที่ยังขาด
        5. ใช้ภาษาธรรมชาติ อบอุ่น คุยง่าย และเป็นกันเอง
        6. คำถามควรช่วยให้ผู้ใช้ตอบต่อได้ง่าย
        7. สามารถยกตัวอย่างสั้น ๆ ได้ ถ้าช่วยให้ตอบง่ายขึ้น
        8. ห้ามสรุปหรือเดาเกินจากที่ผู้ใช้พูด
        9. ถ้า focus = learning_need ให้ถามในมุมว่า "ตอนนี้อยากให้ช่วยเรื่องไหนมากที่สุด"
        10. คำถามควรยาวพอประมาณ ไม่สั้นห้วนเกินไป แต่ยังอ่านลื่นและฟังเป็นธรรมชาติ

        แนวทางสำคัญ:
        - ถ้า focus = topic → ให้ถามว่า "อยากพัฒนาเรื่องอะไรเป็นหลัก"
        - ถ้า focus = goal → ให้ถามว่า "อยากทำอะไรได้ดีขึ้น"
        - ถ้า focus = event → ให้ถามว่า "จะนำไปใช้ในสถานการณ์ไหน"
        - ถ้า focus = learning_need → ให้ถามว่า "ตอนนี้อยากให้ช่วยเรื่องไหนมากที่สุด"

        ข้อห้าม:
        - ห้ามตอบยาวเป็นย่อหน้า
        - ห้ามถามหลายคำถามซ้อน
        - ห้ามใช้โทนแข็งหรือเป็นระบบเกินไป
        - ห้ามเสนอหลายตัวเลือกมากเกินจำเป็น
        - ห้ามตอบเป็นข้อ ๆ
        - ห้ามอธิบายเยอะก่อนถาม

        {examples}

        รูปแบบคำตอบ:
        - ตอบเป็นภาษาไทย
        - 1 ประโยค หรือไม่เกิน 2 ประโยค
        - คำถามควรยาวพอประมาณและเป็นธรรมชาติ
        - ต้องเป็นคำถามที่ชวนให้ผู้ใช้ตอบต่อได้ง่าย
        - ตอบเป็น “ข้อความคำถาม” เท่านั้น ไม่มีคำนำ ไม่มี bullet ไม่มี quote
        """

    user_prompt = f"""คำถามล่าสุดของโค้ช: {last_question}
        ข้อความล่าสุดของผู้ใช้: {user_message}

        ช่วยสร้างคำถามถัดไปที่ต่อเนื่องจากข้อความนี้ โดยถามเฉพาะข้อมูลที่ยังขาดอยู่"""

    result = await call_openai_chat_full(
    model=model,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    temperature=0.6,
    )

    content = (result["content"] or "").strip()

    if not content:
        content = "ช่วยเล่าเพิ่มเติมอีกนิดได้ไหมครับ"

    return content

def build_query_text_by_ai(topic: str, goal: str, event: str) -> dict:
    system = """คุณคือ AI ที่ช่วยสร้าง query text สำหรับใช้ค้นหาข้อมูลใน vector database

    กติกา:
    - รับ topic, goal และ event
    - สร้างข้อความค้นหา 1 ย่อหน้าสั้น ๆ
    - เน้น keyword สำคัญที่เกี่ยวกับหัวข้อและความต้องการเรียนรู้
    - ไม่ต้องตอบยาว
    - ไม่ต้องใส่ bullet
    - ไม่ต้องอธิบายเกินความจำเป็น"""

    user = (
        f"topic: {topic}\n"
        f"goal: {goal}\n"
        f"event: {event}\n\n"
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
            "query_text": f"หัวข้อ {topic} goal {goal} event {event}",
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
    goal: str,
    event: str,
    query_text: str,
    context: str,
    follow: str,
    ) -> dict:

    system = """คุณคือ AI Coach ที่ช่วยตอบผู้ใช้จากข้อมูล knowledge ที่มีอยู่

    กติกา:
    - ใช้ "Context" เท่านั้นในการตอบเนื้อหาหลัก
    - ห้ามนำ "Topic ที่เกี่ยวข้อง" มาใช้เป็นสาระหลักของคำตอบ
    - "Topic ที่เกี่ยวข้อง" ใช้ได้เฉพาะช่วงท้าย เพื่อชวนผู้ใช้ต่อยอดบทสนทนา
    - ตอบให้สัมพันธ์กับ topic, goal, และ event
    - หาก Context มีหลายเรื่อง ให้เลือกเฉพาะส่วนที่เกี่ยวข้องมากที่สุด
    - อธิบายให้เข้าใจง่าย กระชับ และนำไปใช้ได้จริง
    - หากข้อมูลยังไม่พอ ให้บอกอย่างตรงไปตรงมาว่าพบข้อมูลใกล้เคียง แต่ยังไม่เพียงพอ
    - ห้ามแต่งรายละเอียดเกินจาก Context
    - ใช้โทนเหมือนโค้ชคุยกับผู้ใช้ ไม่ให้แข็งหรือเหมือนระบบมากเกินไป
    - จัด format สำหรับแสดงใน HTML โดยใช้แท็กเรียบง่าย เช่น div, p, ul, li, strong

    โครงสร้างคำตอบที่ต้องการ:
    1. เริ่มจาก "แนะนำเนื้อหาความรู้ที่เกี่ยวข้อง" ก่อน
    - บอกสั้น ๆ ว่าจากข้อมูลที่มี เนื้อหานี้เกี่ยวข้องกับสิ่งที่ผู้ใช้กำลังต้องการพัฒนาอย่างไร
    - สรุปสาระสำคัญจาก Context แบบเข้าใจง่าย
    - ถ้ามีหลายข้อ ให้สรุปเป็น bullet ที่อ่านง่าย

    2. จากนั้น "สรุปผลลัพธ์หรือประโยชน์" ที่ผู้ใช้น่าจะได้รับ
    - บอกว่าเมื่อเข้าใจหรือฝึกตามเนื้อหานี้แล้ว จะช่วยให้ผู้ใช้ดีขึ้นอย่างไร
    - ต้องเชื่อมกับ goal และ event ของผู้ใช้
    - เน้นผลลัพธ์เชิงปฏิบัติ เช่น ทำได้ชัดขึ้น มั่นใจขึ้น จัดการได้ดีขึ้น เป็นต้น

    3. ตอนท้าย "ชวนต่อยอด" จาก Topic ที่เกี่ยวข้อง
    - ใช้ Topic ที่เกี่ยวข้องได้เฉพาะช่วงท้าย
    - ชวนอย่างเป็นธรรมชาติ ไม่ยัดเยียด
    - ห้ามใช้ Topic ที่เกี่ยวข้องมาเป็นคำตอบหลัก

    ข้อห้าม:
    - ห้ามเปิดคำตอบด้วยการวิเคราะห์ผู้ใช้ยาว ๆ
    - ห้ามเริ่มจากการบอกว่าเขาขาด competency อะไรเป็นหลัก
    - ห้ามตอบเป็นทฤษฎียาวเกินไป
    - ห้ามเอา Topic ที่เกี่ยวข้องมาใช้แทน Context
    - ห้ามใช้ข้อมูลนอก Context

    รูปแบบคำตอบ:
    - ตอบเป็น HTML แบบเรียบง่าย
    - ใช้ <div>, <p>, <ul>, <li>, <strong> เท่าที่จำเป็น
    - โครงอ่านง่าย เป็นธรรมชาติ
    """

    user = f"""คำถามผู้ใช้: {user_message}

    Topic: {topic}
    Goal: {goal}
    Event: {event}
    Query Text ที่ใช้ค้นหา: {query_text}

    Context สำหรับใช้ตอบคำถามหลัก:
    {context}

    Topic ที่เกี่ยวข้อง (ใช้เฉพาะสำหรับชวนคุยต่อท้ายคำตอบ ไม่ใช้เป็นข้อมูลหลัก):
    {follow}

    ช่วยตอบแบบเป็นธรรมชาติ เหมือนโค้ชกำลังคุยกับผู้ใช้จริง ไม่ให้ดูเหมือนการสรุปเป็นหัวข้อหรือรายงาน

    แนวทางในการตอบ:
    - เริ่มจากเชื่อมให้เห็นก่อนว่าเนื้อหาใน Context นี้เกี่ยวข้องกับสิ่งที่ผู้ใช้กำลังอยากพัฒนาอย่างไร
    - จากนั้นค่อยอธิบายสาระสำคัญที่ relevant ที่สุดจาก Context แบบเนียน ๆ ในบทสนทนา
    - แล้วเชื่อมต่อไปสู่สิ่งที่ผู้ใช้น่าจะได้หรือผลลัพธ์ที่น่าจะเกิดขึ้น หากนำแนวคิดนี้ไปใช้
    - ตอนท้ายค่อยชวนต่อยอดจาก Topic ที่เกี่ยวข้องอย่างเป็นธรรมชาติ เหมือนชวนคุยต่อ ไม่ใช่เสนอขายหรือยัดเยียด

    ข้อสำคัญ:
    - อย่าตอบแบบแบ่งย่อหน้าตามหัวข้อชัดเจนเกินไป
    - อย่าใช้คำสไตล์สรุป เช่น “สาระสำคัญคือ”, “ผลลัพธ์ที่ได้คือ”, “สรุปคือ”
    - ให้ลื่นเหมือนคนอธิบายต่อเนื่องกัน
    - ให้ใส่ tag html คำสำคัญหรือ keyword ให้สวยงาม
    - ใช้ภาษาที่อบอุ่น เข้าใจง่าย และนำไปใช้ได้จริง

    หาก Context ไม่พอ ให้บอกตรง ๆ ว่าพบข้อมูลใกล้เคียง แต่ยังไม่เพียงพอ และยังคงตอบตามโครงสร้างเดิมเท่าที่ทำได้"""

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
    goal: str,
    event: str,
    learning_need: str,
) -> dict:
    
    # 1) ให้ AI สร้าง query text
    q = build_query_text_by_ai(topic, goal, event)
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
        goal=goal,
        event=event,
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