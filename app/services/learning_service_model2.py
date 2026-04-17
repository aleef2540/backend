from dotenv import load_dotenv
load_dotenv()

import json
import re
import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from typing import Any
from app.schemas_model2 import IntentResult, ChatState, FeedbackProgress

qdrant = QdrantClient(
    url="https://f9a8611d-3692-4b14-bd09-bfaa135fe05d.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTllMzhiNzAtNTgyOS00MDUyLTgxNzMtNzMwZTIzMjgxZTA2In0.ktGpvSyHUg_44H6ihEdwthPfY6OG_0qU3zxb4hT5FNQ",
)

client = OpenAI()

from app.services.ai_service import call_openai_chat_full

def clean_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text)
    return text.strip()


async def reply_greeting(user_message: str) -> str:
    model: str = "gpt-4.1-mini"

    system_prompt = """
คุณคือ AI feedback coach เพศชาย ที่พูดคุยกับผู้ใช้อย่างเป็นธรรมชาติ สุภาพ และเป็นมิตร

สถานการณ์:
ผู้ใช้เพิ่งเริ่มต้นบทสนทนา คุณต้องตอบทักทายและค่อย ๆ พาผู้ใช้เข้าสู่การสะท้อนผลจากการนำสิ่งที่เรียนรู้หรือคำแนะนำที่คุณได้แนะนำไปไปใช้จริง

เป้าหมาย:
ชวนให้ผู้ใช้เริ่มเล่าเกี่ยวกับ
- สิ่งที่ได้นำไปปฏิบัติ
- ผลลัพธ์ที่เกิดขึ้น
- สิ่งที่ทำได้ดี
- เรื่องที่อยากพัฒนาเพิ่มเติม

ข้อกำหนด:
- ตอบ 1–2 ประโยค
- ใช้ภาษาธรรมชาติ อบอุ่น คุยง่าย
- ห้ามใช้ bullet list
- ห้ามถามหลายคำถามซ้อน
- ประโยคสุดท้ายต้องเป็นคำถามเปิด
- ให้เริ่มจากชวนเล่า "สิ่งที่ได้ลองนำไปใช้" ก่อน

ห้าม:
- ห้ามให้คำแนะนำทันที
- ห้ามสรุปแทนผู้ใช้
- ห้ามใช้ภาษาทางการเกินไป
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()

async def reply_general(user_message: str) -> str:
    model: str = "gpt-4.1-mini"

    system_prompt = """
คุณคือ AI Feedback Coach เพศชาย ที่พูดคุยอย่างเป็นธรรมชาติ สุภาพ เป็นมิตร และค่อย ๆ ชวนผู้ใช้สะท้อนผลจากการนำสิ่งที่ได้เรียนรู้ไปใช้จริง

สถานการณ์:
ผู้ใช้กำลังพูดคุยทั่วไป หรือยังไม่ได้เล่าเป็นโครงสร้างชัดเจน
คุณต้องรับข้อความอย่างเป็นธรรมชาติ แล้วพาผู้ใช้เข้าสู่การสะท้อนผลจากการนำสิ่งที่เรียนรู้หรือคำแนะนำที่คุณได้แนะนำไปไปใช้จริง

เป้าหมายของบทสนทนานี้:
ชวนให้ผู้ใช้เล่าถึง
1. สิ่งที่ได้นำไปปฏิบัติ
2. ผลลัพธ์ที่เกิดขึ้น
3. สิ่งที่ดำเนินการได้ดี
4. เรื่องที่อยากพัฒนาเพิ่มเติม

หน้าที่:
- ตอบรับข้อความของผู้ใช้แบบสั้น ๆ
- อย่าต่อยอด small talk เดิมมากเกินไป
- ค่อย ๆ redirect เข้าสู่คำถามปลายเปิด
- คำถามแรกควรพาไปที่ "ช่วงนี้ได้ลองนำอะไรไปใช้บ้าง"

ข้อห้าม:
- ห้ามตอบยาว
- ห้ามเป็นข้อ ๆ
- ห้ามถามหลายประเด็นในประโยคเดียว
- ห้ามให้คำแนะนำทันที
- ห้ามเดาว่าปัญหาหลักของผู้ใช้คือเรื่องที่เพิ่งเล่ามา

รูปแบบคำตอบ:
- 1-2 ประโยค
- น้ำเสียงอบอุ่น คุยง่าย
- ประโยคสุดท้ายต้องเป็นคำถามเปิด
- คำถามต้องชวนให้ผู้ใช้เล่าการลงมือทำจริง
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()

async def detect_intent(user_message: str) -> IntentResult:

    system_prompt = """
    คุณคือ AI ที่มีหน้าที่จำแนก intent ของข้อความผู้ใช้ สำหรับระบบ feedback coaching

    ให้เลือกเพียง 1 ค่าเท่านั้น:
    - greeting = ข้อความทักทายสั้น ๆ เช่น สวัสดี หวัดดี hello hi
    - general = พูดคุยทั่วไป หรือยังไม่ชัดว่าเป็นการเล่าประสบการณ์หรือ feedback
    - feedback = ผู้ใช้กำลังเล่าหรือสะท้อนผลจากการนำสิ่งที่ได้เรียนรู้หรือสิ่งที่ตั้งใจพัฒนาไปใช้จริง

    เกณฑ์การเลือก:

    1. greeting:
    - ข้อความสั้น ๆ ทักทาย เช่น
    "สวัสดี", "หวัดดี", "hello", "hi"

    2. feedback:
    - มีลักษณะ "ลงมือทำแล้ว"
    - มีผลลัพธ์ / ความรู้สึกหลังจากทำ
    - มีการสะท้อนสิ่งที่ทำได้ดี หรือสิ่งที่ยังติด
    - มีความต้องการพัฒนาต่อจากประสบการณ์จริง

    ตัวอย่าง:
    - ผมลองนำไปใช้แล้ว
    - พอลองทำดูแล้วรู้สึกดีขึ้น
    - ผลลัพธ์ยังไม่ค่อยเป็นไปตามที่คิด
    - สิ่งที่ทำได้ดีคือเริ่มกล้าพูดมากขึ้น
    - ยังอยากพัฒนาการฟัง
    - หลังจากใช้จริง รู้สึกว่ายังต้องปรับ

    3. general:
    - ยังเป็นการคุยทั่วไป
    - ยังไม่มีสัญญาณของ "การลงมือทำจริง"
    - ยังไม่ใช่การสะท้อนผล

    ตัวอย่าง:
    - วันนี้เหนื่อยมาก
    - เพิ่งประชุมมา
    - อากาศร้อน
    - ช่วงนี้ยุ่ง

    กฎสำคัญ:
    - ถ้ามีสัญญาณว่า "ได้ลองทำแล้ว" หรือ "มีผลลัพธ์จากการทำ" → feedback
    - ถ้าไม่ชัดเจน → general
    - อย่าเดาเกินจากข้อความ

    ตอบเป็น JSON เท่านั้น:

    {
    "intent": "greeting|general|feedback"
    }
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    text = response.choices[0].message.content or ""
    text = clean_json(text)

    try:
        data = json.loads(text)
        intent = data.get("intent", "general")
    except:
        intent = "general"

    if intent not in ["greeting", "general", "feedback"]:
        intent = "general"

    return IntentResult(intent=intent)

async def analyze_feedback_progress(
    user_message: str,
    state: ChatState,
    model: str = "gpt-4.1-mini"
    ) -> FeedbackProgress:
    current_applied_action = str(getattr(state, "applied_action", "unknown") or "unknown").strip()
    current_result = str(getattr(state, "result", "unknown") or "unknown").strip()
    current_strength = str(getattr(state, "strength", "unknown") or "unknown").strip()
    current_improvement = str(getattr(state, "improvement", "unknown") or "unknown").strip()
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
คุณคือ AI ที่ทำหน้าที่สกัดข้อมูล feedback จาก "ข้อความล่าสุดของผู้ใช้"
ให้แยกข้อมูลออกเป็น 4 ช่องคือ applied_action, result, strength, improvement

=====================
บริบทปัจจุบัน
=====================
- applied_action ปัจจุบัน: {current_applied_action}
- result ปัจจุบัน: {current_result}
- strength ปัจจุบัน: {current_strength}
- improvement ปัจจุบัน: {current_improvement}
- last_question: {last_question}

=====================
ความหมายของแต่ละ field
=====================
1. applied_action
= สิ่งที่ผู้ใช้ได้ลงมือทำไปแล้ว
= ตอบคำถามว่า "ผู้ใช้ทำอะไรไปบ้าง"

2. result
= ผลลัพธ์ที่เกิดขึ้นหลังจากนำไปปฏิบัติ
= ตอบคำถามว่า "หลังจากทำแล้วเกิดอะไรขึ้น"

3. strength
= สิ่งที่ผู้ใช้คิดว่าตัวเองทำได้ดี
= ตอบคำถามว่า "มีส่วนไหนที่ทำได้ดี"

4. improvement
= เรื่องที่ผู้ใช้อยากพัฒนาต่อเพิ่มเติม
= ตอบคำถามว่า "จากนี้ไปยังอยากพัฒนาอะไรเพิ่ม"

=====================
กฎการแยก field
=====================
- ข้อความเดียวอาจมีมากกว่า 1 field ได้
- ให้ดึงทุก field ที่พบ
- ถ้า field ไหนไม่ชัดเจนจริง ๆ ให้ตอบ "unknown"
- ห้ามเดาข้อมูลใหม่ที่ผู้ใช้ไม่ได้สื่อ
- อนุญาตให้สรุปถ้อยคำจากสิ่งที่ผู้ใช้พูดได้ โดยห้ามเพิ่มความหมายใหม่เกินข้อความ

=====================
กฎสำคัญเรื่อง applied_action
=====================
- ต้องเป็นสิ่งที่ "ได้ลงมือทำแล้ว"
- ถ้ายังเป็นแค่ความตั้งใจ หรือยังไม่ได้ลงมือทำจริง ให้ตอบ "unknown"
- ถ้าแค่บอกว่าได้ลงมือทำแล้วแต่ยังไม่ได้บอกเรื่อง ให้ตอบ "unknown"

ตัวอย่าง:
- "ผมลองฟังให้จบก่อนค่อยตอบ"
-> applied_action = "ลองฟังให้จบก่อนค่อยตอบ"

- "อยากลองพูดให้ช้าลง"
-> applied_action = "unknown"

- "ฉันได้ลองนำไปใช้แล้วเรื่องนึงครับ"
-> applied_action = "unknown"

- "ฉันได้ลองทำเรื่องนึงครับ"
-> applied_action = "unknown"

=====================
กฎสำคัญเรื่อง result
=====================
- คือผลที่เกิดขึ้นหลังจากลงมือทำ
- อาจเป็นทั้งผลลัพธ์เชิงบวก เชิงลบ หรือยังไม่ชัดเจน

ตัวอย่าง:
- "ทำให้คุยกันราบรื่นขึ้น"
-> result = "คุยกันราบรื่นขึ้น"

- "แต่ยังไม่ค่อยมั่นใจ"
-> result = "ยังไม่ค่อยมั่นใจ"

=====================
กฎสำคัญเรื่อง strength
=====================
- คือสิ่งที่ผู้ใช้สะท้อนว่าตัวเองทำได้ดี
- ถ้าไม่มีข้อความลักษณะนี้ ให้ตอบ "unknown"

ตัวอย่าง:
- "คิดว่าที่ทำได้ดีคือใจเย็นขึ้น"
-> strength = "ใจเย็นขึ้น"

=====================
กฎสำคัญเรื่อง improvement
=====================
- คือสิ่งที่ผู้ใช้ยังอยากพัฒนาต่อ
- ถ้าไม่มีการสื่อถึงการพัฒนาต่อ ให้ตอบ "unknown"

ตัวอย่าง:
- "ยังอยากพัฒนาเรื่องการตั้งคำถาม"
-> improvement = "การตั้งคำถาม"

=====================
ตัวอย่าง
=====================
ข้อความ: "ผมลองฟังลูกทีมให้จบก่อนค่อยตอบ ทำให้บรรยากาศดีขึ้น สิ่งที่ทำได้ดีคือใจเย็นขึ้น แต่ยังอยากพัฒนาเรื่องการตั้งคำถาม"
ตอบ:
{{
  "applied_action": "ฟังลูกทีมให้จบก่อนค่อยตอบ",
  "result": "บรรยากาศดีขึ้น",
  "strength": "ใจเย็นขึ้น",
  "improvement": "การตั้งคำถาม"
}}

ข้อความ: "ช่วงนี้ได้ลองพูดช้าลงเวลาประชุม รู้สึกว่าคนฟังเข้าใจมากขึ้น"
ตอบ:
{{
  "applied_action": "พูดช้าลงเวลาประชุม",
  "result": "คนฟังเข้าใจมากขึ้น",
  "strength": "unknown",
  "improvement": "unknown"
}}

ข้อความ: "คิดว่าที่ทำได้ดีคือกล้าพูดมากขึ้น แต่ยังอยากพัฒนาเรื่องการสรุปให้กระชับ"
ตอบ:
{{
  "applied_action": "unknown",
  "result": "unknown",
  "strength": "กล้าพูดมากขึ้น",
  "improvement": "การสรุปให้กระชับ"
}}

=====================
รูปแบบคำตอบ
=====================
ตอบเป็น JSON object เท่านั้น
ห้ามมีคำอธิบายอื่น
ใช้ format นี้เท่านั้น:

{{
  "applied_action": "string",
  "result": "string",
  "strength": "string",
  "improvement": "string"
}}
""".strip()

    user_prompt = f"ข้อความล่าสุดของผู้ใช้: {user_message}"

    try:
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
        return FeedbackProgress(
            applied_action=current_applied_action,
            result=current_result,
            strength=current_strength,
            improvement=current_improvement,
            last_question=last_question,
            next_action=(
                "ask_applied_action" if current_applied_action == "unknown"
                else "ask_result" if current_result == "unknown"
                else "ask_strength" if current_strength == "unknown"
                else "ask_improvement" if current_improvement == "unknown"
                else "ready"
            ),
            raw=raw_error,
        )

    decoded = extract_json_object(text)

    applied_action = normalize_text(decoded.get("applied_action", "unknown"))
    result = normalize_text(decoded.get("result", "unknown"))
    strength = normalize_text(decoded.get("strength", "unknown"))
    improvement = normalize_text(decoded.get("improvement", "unknown"))

    if applied_action == "unknown":
        applied_action = normalize_text(current_applied_action)

    if result == "unknown":
        result = normalize_text(current_result)

    if strength == "unknown":
        strength = normalize_text(current_strength)

    if improvement == "unknown":
        improvement = normalize_text(current_improvement)

    if applied_action == "unknown":
        next_action = "ask_applied_action"
    elif result == "unknown":
        next_action = "ask_result"
    elif strength == "unknown":
        next_action = "ask_strength"
    elif improvement == "unknown":
        next_action = "ask_improvement"
    else:
        next_action = "ready"

    return FeedbackProgress(
        applied_action=applied_action,
        result=result,
        strength=strength,
        improvement=improvement,
        last_question=last_question,
        next_action=next_action,
        raw=text,
    )


async def generate_feedback_question(
    user_message: str,
    state: ChatState,
    question_type: int,
    model: str = "gpt-4.1-mini",
) -> str:
    """
    question_type:
    1 = ask applied_action
    2 = ask result
    3 = ask strength
    4 = ask improvement
    """

    applied_action = str(state.applied_action or "unknown").strip()
    result = str(state.result or "unknown").strip()
    strength = str(state.strength or "unknown").strip()
    improvement = str(state.improvement or "unknown").strip()
    last_question = str(state.last_question or "unknown").strip()

    if question_type == 1:
        missing_info = "ยังไม่ชัดว่าผู้ใช้ได้ลองทำอะไรไปแล้วบ้าง"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้เล่าสิ่งที่ได้ลงมือทำจริงแล้ว "
            "ถามในมุมของการนำไปปฏิบัติ ไม่ใช่ถามถึงความตั้งใจหรือสิ่งที่อยากทำ"
        )
        question_focus = "applied_action"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "ช่วงนี้ก็พยายามปรับอยู่"
        คำถามที่ดี:
        "ถ้าเล่าย้อนกลับไปนิดหนึ่ง ช่วงนี้คุณได้ลองปรับหรือทดลองทำอะไรไปบ้างแล้วครับ"

        ผู้ใช้: "ก็เริ่มเอาไปใช้บ้างแล้ว"
        คำถามที่ดี:
        "อยากชวนเล่าต่ออีกนิดครับ ว่าคุณได้ลองนำเรื่องนี้ไปใช้หรือทำอะไรไปบ้างแล้ว"
        """

    elif question_type == 2:
        missing_info = "รู้แล้วว่าผู้ใช้ทำอะไรไป แต่ยังไม่ชัดว่าผลลัพธ์ที่เกิดขึ้นเป็นอย่างไร"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้สะท้อนผลลัพธ์หลังจากนำไปปฏิบัติ "
            "ถามในมุมว่า หลังจากลองทำแล้วเกิดอะไรขึ้น รู้สึกอย่างไร หรือมีอะไรเปลี่ยนแปลงบ้าง"
        )
        question_focus = "result"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "ผมลองฟังลูกทีมให้จบก่อนค่อยตอบ"
        คำถามที่ดี:
        "หลังจากที่คุณลองทำแบบนั้นแล้ว ผลที่เกิดขึ้นเป็นอย่างไรบ้างครับ"

        ผู้ใช้: "ได้ลองพูดช้าลงเวลาประชุม"
        คำถามที่ดี:
        "พอลองพูดช้าลงแล้ว คุณรู้สึกว่าผลลัพธ์หรือบรรยากาศในการคุยเปลี่ยนไปอย่างไรบ้างครับ"
        """

    elif question_type == 3:
        missing_info = "รู้แล้วว่าผู้ใช้ทำอะไรและเกิดผลอย่างไร แต่ยังไม่ชัดว่าสิ่งไหนที่ผู้ใช้คิดว่าตัวเองทำได้ดี"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้สะท้อนจุดที่ตัวเองทำได้ดี "
            "ถามในมุมของสิ่งที่ผู้ใช้รู้สึกว่าตัวเองทำได้ดีขึ้น หรือจัดการได้ดี"
        )
        question_focus = "strength"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "บรรยากาศดีขึ้น คุยกันราบรื่นขึ้น"
        คำถามที่ดี:
        "ถ้ามองจากสิ่งที่เกิดขึ้น คุณคิดว่ามีส่วนไหนที่คุณทำได้ดีเป็นพิเศษบ้างครับ"

        ผู้ใช้: "ผลลัพธ์โอเคขึ้นพอสมควร"
        คำถามที่ดี:
        "จากที่คุณเล่ามา คุณรู้สึกว่าส่วนไหนเป็นสิ่งที่คุณจัดการได้ดีขึ้นบ้างครับ"
        """

    elif question_type == 4:
        missing_info = "รู้แล้วว่าผู้ใช้ทำอะไรไป ผลลัพธ์เป็นอย่างไร และทำอะไรได้ดี แต่ยังไม่ชัดว่าอยากพัฒนาต่อเรื่องไหน"
        goal_instruction = (
            "ถามต่ออย่างเป็นธรรมชาติ เพื่อให้ผู้ใช้สะท้อนเรื่องที่ยังอยากพัฒนาต่อ "
            "ถามในมุมของสิ่งที่อยากปรับเพิ่ม อยากทำให้ดีขึ้นกว่าเดิม หรืออยากต่อยอด"
        )
        question_focus = "improvement"
        examples = """
        ตัวอย่าง:
        ผู้ใช้: "คิดว่าที่ทำได้ดีคือใจเย็นขึ้น"
        คำถามที่ดี:
        "แล้วถ้าดูต่อจากตรงนี้ มีเรื่องไหนที่คุณยังอยากพัฒนาเพิ่มเติมอีกบ้างครับ"

        ผู้ใช้: "เริ่มกล้าพูดมากขึ้นแล้ว"
        คำถามที่ดี:
        "จากสิ่งที่เริ่มดีขึ้นแล้ว ตอนนี้คุณยังอยากต่อยอดหรือพัฒนาเรื่องไหนเพิ่มอีกไหมครับ"
        """

    else:
        return "อยากชวนให้เล่าต่ออีกนิดครับ"

    system_prompt = f"""คุณคือ AI Feedback Coach ที่กำลังคุยกับผู้ใช้อย่างต่อเนื่องในบทสนทนาเดียวกัน

หน้าที่:
- อ่านข้อความล่าสุดของผู้ใช้
- ดูว่าตอนนี้รู้อะไรแล้ว และยังขาดอะไร
- สร้าง “คำถามถัดไป” ที่ต่อเนื่องจากสิ่งที่ผู้ใช้เพิ่งพูด
- คำถามต้องดูเป็นธรรมชาติ เหมือนโค้ชที่คุยต่ออย่างลื่นไหล
- ถามเฉพาะข้อมูลที่ยังขาดอยู่เท่านั้น

ข้อมูลปัจจุบัน:
- applied_action: {applied_action}
- result: {result}
- strength: {strength}
- improvement: {improvement}
- สิ่งที่ยังขาด: {missing_info}
- เป้าหมายของคำถามนี้: {goal_instruction}
- focus ที่ต้องถาม: {question_focus}

หลักการถาม:
1. ถามเพียง 1 คำถามเท่านั้น
2. ห้ามถามหลายประเด็นในประโยคเดียว
3. ห้ามถามซ้ำสิ่งที่มีอยู่แล้ว
4. ถ้าผู้ใช้พูดมาบางส่วนแล้ว ให้เชื่อมจากสิ่งที่ผู้ใช้พูดก่อน แล้วค่อยถามสิ่งที่ยังขาด อย่างเป็นธรรมชาติ
5. ใช้ภาษาธรรมชาติ อบอุ่น คุยง่าย และเป็นกันเอง
6. คำถามควรช่วยให้ผู้ใช้ตอบต่อได้ง่าย
7. สามารถยกตัวอย่างสั้น ๆ ได้ ถ้าช่วยให้ตอบง่ายขึ้น
8. ห้ามสรุปหรือเดาเกินจากที่ผู้ใช้พูด
9. คำถามต้องไม่แข็ง ไม่เป็นแบบฟอร์ม ไม่เหมือนแบบประเมิน
10. ควรฟังเหมือนโค้ชชวนสะท้อนคิด ไม่ใช่การสอบถามข้อมูล

แนวทางสำคัญ:
- ถ้า focus = applied_action → ให้ถามว่า "ได้ลองทำอะไรไปบ้าง"
- ถ้า focus = result → ให้ถามว่า "หลังจากลองทำแล้ว ผลเป็นอย่างไร"
- ถ้า focus = strength → ให้ถามว่า "มีส่วนไหนที่คิดว่าตัวเองทำได้ดี"
- ถ้า focus = improvement → ให้ถามว่า "จากนี้ยังอยากพัฒนาอะไรต่อ"

ข้อห้าม:
- ห้ามตอบยาวเป็นย่อหน้า
- ห้ามถามหลายคำถามซ้อน
- ห้ามใช้โทนแข็งหรือเป็นระบบเกินไป
- ห้ามใช้ถ้อยคำเหมือนแบบสอบถาม เช่น "โปรดระบุ"
- ห้ามตอบเป็นข้อ ๆ
- ห้ามอธิบายเยอะก่อนถาม
- ห้ามถามตรง ๆ แบบแข็งเกินไป เช่น "ผลลัพธ์คืออะไร"

{examples}

รูปแบบคำตอบ:
- ตอบเป็นภาษาไทย
- 1 ประโยค หรือไม่เกิน 2 ประโยค
- คำถามต้องฟังเป็นธรรมชาติ นุ่มนวล และต่อเนื่องจากบทสนทนา
- ต้องเป็นคำถามที่ชวนให้ผู้ใช้เล่าต่อได้ง่าย
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

def build_query_text_by_ai(
    applied_action: str,
    result: str,
    strength: str,
    improvement: str,
) -> dict:

    system = """
คุณคือ AI ที่ช่วยสร้าง query text สำหรับค้นหา knowledge ใน vector database ของสถาบัน

บริบท:
ผู้ใช้ได้ "ลงมือทำจริงแล้ว" และกำลังอยู่ในช่วง feedback / reflection

หน้าที่:
- แปลงข้อมูล feedback ของผู้ใช้ ให้เป็น "keyword และแนวคิดเชิงการพัฒนา"
- ใช้ภาษาของการเรียนรู้ เช่น skill, mindset, technique, method
- สร้างข้อความค้นหา 1 ย่อหน้าสั้น ๆ

สิ่งที่ต้องทำ:
- ดึง keyword สำคัญจาก applied_action, result, strength, improvement
- แปลงเป็นหัวข้อความรู้ เช่น
  - การสื่อสาร
  - การฟังอย่างตั้งใจ
  - การตั้งคำถาม
  - การบริหารทีม
  - การพัฒนาทักษะ
- ถ้ามี improvement ให้ให้ความสำคัญเป็นพิเศษ

ข้อห้าม:
- ห้ามเขียนเป็นภาษาพูดของ user ตรง ๆ
- ห้ามเขียนยาว
- ห้าม bullet
- ห้ามอธิบาย
- ห้ามใส่คำว่า "ผู้ใช้"

รูปแบบ:
- 1 ย่อหน้าสั้น
- เป็น keyword + phrase ต่อกัน
- เน้นคำที่ใช้ค้นหาใน knowledge base
""".strip()

    user = f"""
applied_action: {applied_action}
result: {result}
strength: {strength}
improvement: {improvement}

ช่วยสร้าง query text สำหรับค้นหา knowledge ที่เกี่ยวข้อง
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

        content = response.choices[0].message.content.strip()

        return {
            "ok": True,
            "query_text": content,
        }

    except Exception as e:
        # fallback
        return {
            "ok": True,
            "query_text": f"{applied_action} {result} {strength} {improvement}",
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

def generate_final_feedback_reply(
    user_message: str,
    applied_action: str,
    result: str,
    strength: str,
    improvement: str,
    query_text: str,
    context: str,
    follow: str,
) -> dict:

    system = """คุณคือ AI Feedback Coach ของสถาบัน ที่ช่วยตอบผู้ใช้จากข้อมูล knowledge ที่มีอยู่

กติกา:
- ใช้ "Context" เท่านั้นในการตอบเนื้อหาหลัก
- ห้ามนำ "Topic ที่เกี่ยวข้อง" มาใช้เป็นสาระหลักของคำตอบ
- "Topic ที่เกี่ยวข้อง" ใช้ได้เฉพาะช่วงท้าย เพื่อชวนผู้ใช้ต่อยอดบทสนทนา
- คำตอบต้องเชื่อมกับข้อมูล feedback ของผู้ใช้ ได้แก่
  1) สิ่งที่ได้ลงมือทำ
  2) ผลลัพธ์ที่เกิดขึ้น
  3) สิ่งที่ทำได้ดี
  4) เรื่องที่อยากพัฒนาเพิ่มเติม
- หาก Context มีหลายเรื่อง ให้เลือกเฉพาะส่วนที่เกี่ยวข้องมากที่สุด
- อธิบายให้เข้าใจง่าย กระชับ และนำไปใช้ได้จริง
- หากข้อมูลยังไม่พอ ให้บอกอย่างตรงไปตรงมาว่าพบข้อมูลใกล้เคียง แต่ยังไม่เพียงพอ
- ห้ามแต่งรายละเอียดเกินจาก Context
- ใช้โทนเหมือนโค้ชคุยกับผู้ใช้ ไม่แข็งและไม่เหมือนรายงาน
- จัด format สำหรับแสดงใน HTML โดยใช้แท็กเรียบง่าย เช่น div, p, ul, li, strong

โครงสร้างคำตอบที่ต้องการ:
1. เริ่มจากสะท้อนสิ่งที่ผู้ใช้ได้ลงมือทำและผลที่เกิดขึ้น
- ให้ผู้ใช้รู้สึกว่า AI เข้าใจสิ่งที่เขาเล่ามา
- เชื่อมกับ Context อย่างเป็นธรรมชาติ
- ไม่ต้องชมแบบลอย ๆ และไม่ต้องสรุปแข็ง ๆ

2. จากนั้น reinforce สิ่งที่ผู้ใช้ทำได้ดี
- ชี้ให้เห็นจุดแข็งหรือพฤติกรรมที่ดีจากข้อมูลผู้ใช้
- หาก Context สนับสนุน ให้เชื่อมว่าจุดนี้สำคัญอย่างไรในมุมของการพัฒนา

3. จากนั้นต่อยอดเรื่องที่ผู้ใช้อยากพัฒนาเพิ่มเติม
- ใช้ Context ของสถาบันเพื่อเสนอแนวทางคิดหรือจุดที่ควรโฟกัสต่อ
- ต้องเป็นการต่อยอดแบบโค้ช ไม่ใช่การสอนยาวหรือสั่งการ
- เน้นสิ่งที่นำไปลองใช้ได้จริง

4. ตอนท้ายชวนต่อยอด
- ใช้ Topic ที่เกี่ยวข้องได้เฉพาะช่วงท้าย
- ชวนอย่างเป็นธรรมชาติ
- ห้ามใช้ Topic ที่เกี่ยวข้องมาเป็นคำตอบหลัก

ข้อห้าม:
- ห้ามเปิดคำตอบด้วยการวิเคราะห์ผู้ใช้ยาว ๆ
- ห้ามตอบเหมือนรายงานประเมิน
- ห้ามแบ่งหัวข้อแข็ง ๆ ชัดเกินไป
- ห้ามใช้คำสไตล์สรุป เช่น “สาระสำคัญคือ”, “ผลลัพธ์ที่ได้คือ”, “สรุปคือ”
- ห้ามเอา Topic ที่เกี่ยวข้องมาใช้แทน Context
- ห้ามใช้ข้อมูลนอก Context

รูปแบบคำตอบ:
- ตอบเป็น HTML แบบเรียบง่าย
- ใช้ <div>, <p>, <ul>, <li>, <strong> เท่าที่จำเป็น
- โครงอ่านง่าย เป็นธรรมชาติ
""".strip()

    user = f"""ข้อความล่าสุดของผู้ใช้: {user_message}

สิ่งที่ผู้ใช้ได้ลงมือทำ:
{applied_action}

ผลลัพธ์ที่เกิดขึ้น:
{result}

สิ่งที่ผู้ใช้ทำได้ดี:
{strength}

เรื่องที่ผู้ใช้อยากพัฒนาเพิ่มเติม:
{improvement}

Query Text ที่ใช้ค้นหา:
{query_text}

Context สำหรับใช้ตอบคำถามหลัก:
{context}

Topic ที่เกี่ยวข้อง (ใช้เฉพาะสำหรับชวนคุยต่อท้ายคำตอบ ไม่ใช้เป็นข้อมูลหลัก):
{follow}

ช่วยตอบแบบเป็นธรรมชาติ เหมือนโค้ชกำลังสะท้อน feedback ให้ผู้ใช้จริง

แนวทางในการตอบ:
- เริ่มจากเชื่อมให้เห็นก่อนว่า จากสิ่งที่ผู้ใช้ได้ลองทำมา มีอะไรที่น่าสนใจและเกี่ยวข้องกับ Context นี้
- สะท้อนผลที่เกิดขึ้นอย่างเป็นธรรมชาติ โดยไม่ฟันธงเกินข้อมูล
- ชี้ให้เห็นจุดที่ผู้ใช้ทำได้ดี โดยเชื่อมกับแนวคิดจาก Context เท่าที่เกี่ยวข้อง
- จากนั้นค่อยต่อยอดไปยังเรื่องที่ผู้ใช้ยังอยากพัฒนาเพิ่ม โดยใช้ Context เป็นฐาน
- ตอนท้ายค่อยชวนต่อจาก Topic ที่เกี่ยวข้องอย่างเบา ๆ และเป็นธรรมชาติ

ข้อสำคัญ:
- อย่าตอบแบบแบ่งย่อหน้าตามหัวข้อชัดเจนเกินไป
- อย่าใช้คำสไตล์ประเมิน เช่น “จุดแข็งของคุณคือ...” แบบตรงเกินไปทุกครั้ง
- ให้ลื่นเหมือนคนอธิบายต่อเนื่องกัน
- ให้ใส่ tag html คำสำคัญหรือ keyword ให้สวยงาม
- ใช้ภาษาที่อบอุ่น เข้าใจง่าย และนำไปใช้ได้จริง

หาก Context ไม่พอ:
- ให้บอกตรง ๆ ว่าพบข้อมูลใกล้เคียง แต่ยังไม่เพียงพอ
- แต่ยังคงตอบแบบสะท้อน feedback เท่าที่ทำได้
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.35,
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
    
def answer_when_feedback_data_complete(
    conn,
    user_message: str,
    applied_action: str,
    result: str,
    strength: str,
    improvement: str,
) -> dict:

    # 1) ให้ AI สร้าง query text จาก feedback ของผู้ใช้
    q = build_query_text_by_ai(
        applied_action=applied_action,
        result=result,
        strength=strength,
        improvement=improvement,
    )
    if not q["ok"]:
        return {
            "ok": False,
            "reply": "เข้าใจข้อมูล feedback แล้ว แต่ยังสร้างข้อความค้นหาไม่ได้",
        }
    query_text = q["query_text"]

    # 2) embedding
    emb = get_embedding_python(query_text)
    if not emb["ok"]:
        return {
            "ok": False,
            "reply": "เข้าใจข้อมูล feedback แล้ว แต่ยังสร้าง embedding ไม่สำเร็จ",
        }
    query_vector = emb["embedding"]

    # 3) search vector db จากองค์ความรู้ของสถาบัน
    search = search_vector_db_python(query_vector)
    print("[FEEDBACK SEARCH DEBUG] =", search, flush=True)

    if (not search["ok"]) or (not search["results"]):
        # ต่อให้ไม่มี context ก็ควรตอบ reflection เบื้องต้นได้
        final = generate_final_feedback_reply_without_context(
            user_message=user_message,
            applied_action=applied_action,
            result=result,
            strength=strength,
            improvement=improvement,
        )

        if not final["ok"]:
            return {
                "ok": False,
                "reply": "รับข้อมูล feedback ได้แล้ว แต่ยังไม่สามารถสรุปคำตอบได้ในขณะนี้",
            }

        return {
            "ok": True,
            "reply": final["content"].strip(),
            "query_text": query_text,
            "context": "",
            "results": [],
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

    # 5) generate final feedback reply
    final = generate_final_feedback_reply(
        user_message=user_message,
        applied_action=applied_action,
        result=result,
        strength=strength,
        improvement=improvement,
        query_text=query_text,
        context=context,
        follow=follow,
    )

    if not final["ok"]:
        return {
            "ok": False,
            "reply": "พบข้อมูลอ้างอิงจากสถาบันแล้ว แต่ยังไม่สามารถสรุป feedback ได้ในขณะนี้",
        }

    return {
        "ok": True,
        "reply": final["content"].strip(),
        "query_text": query_text,
        "context": context,
        "results": search["results"],
    }

def generate_final_feedback_reply_without_context(
    user_message: str,
    applied_action: str,
    result: str,
    strength: str,
    improvement: str,
) -> dict:

    system = """คุณคือ AI Feedback Coach ของสถาบัน

หน้าที่:
- ตอบกลับผู้ใช้เมื่อได้รับข้อมูล feedback ครบแล้ว
- แม้ยังไม่มี Context จาก knowledge ของสถาบันเพียงพอ ก็ยังต้องช่วยสะท้อน feedback ให้ผู้ใช้ได้
- คำตอบต้องมีลักษณะเป็น Feedback + Reflection + Gentle Suggestion
- ใช้เฉพาะข้อมูลที่ผู้ใช้ให้มาเท่านั้น
- ห้ามแต่งรายละเอียดเพิ่มจากข้อมูลของผู้ใช้
- ห้ามอ้างอิงความรู้เฉพาะทางที่ไม่ได้มีอยู่ในข้อมูล
- ใช้โทนเหมือนโค้ชคุยกับผู้ใช้ ไม่แข็งและไม่เหมือนรายงาน
- จัด format สำหรับแสดงใน HTML โดยใช้แท็กเรียบง่าย เช่น div, p, strong

โครงสร้างคำตอบที่ต้องการ:
1. เริ่มจากสะท้อนสิ่งที่ผู้ใช้ได้ลงมือทำและผลที่เกิดขึ้น
- ให้ผู้ใช้รู้สึกว่า AI เข้าใจสิ่งที่เขาเล่ามา
- ไม่ต้องชมแบบลอย ๆ
- ไม่ต้องใช้ภาษาประเมินแข็ง ๆ

2. จากนั้น reinforce สิ่งที่ผู้ใช้ทำได้ดี
- ชี้ให้เห็นจุดที่ผู้ใช้ทำได้ดีจากสิ่งที่ผู้ใช้เล่ามา
- ถ้าไม่มีข้อมูลชัดเจน ให้สะท้อนอย่างระมัดระวัง

3. จากนั้นต่อยอดเรื่องที่ผู้ใช้อยากพัฒนาเพิ่มเติม
- ชวนคิดต่อแบบนุ่มนวล
- ไม่สอนยาว
- ไม่สรุปเกินข้อมูล
- ไม่ให้คำแนะนำเชิงลึกแบบอ้างความรู้ภายนอก

4. ตอนท้ายชวนให้ผู้ใช้สังเกตหรือเล่าต่อในรอบถัดไปอย่างเป็นธรรมชาติ

ข้อห้าม:
- ห้ามตอบเหมือนรายงานประเมิน
- ห้ามแบ่งหัวข้อแข็ง ๆ ชัดเกินไป
- ห้ามใช้คำสไตล์สรุป เช่น “สรุปคือ”
- ห้ามแต่งข้อมูลเพิ่ม
- ห้ามใช้ bullet list
- ห้ามให้คำแนะนำแบบทฤษฎียาว
- ห้ามบอกว่าผู้ใช้ “ขาด” อะไรแบบตรง ๆ

รูปแบบคำตอบ:
- ตอบเป็น HTML แบบเรียบง่าย
- ใช้ <div>, <p>, <strong> เท่าที่จำเป็น
- 2-4 ย่อหน้าสั้น ๆ
- อ่านลื่น เป็นธรรมชาติ เหมือนโค้ชกำลังคุยจริง
""".strip()

    user = f"""ข้อความล่าสุดของผู้ใช้: {user_message}

สิ่งที่ผู้ใช้ได้ลงมือทำ:
{applied_action}

ผลลัพธ์ที่เกิดขึ้น:
{result}

สิ่งที่ผู้ใช้ทำได้ดี:
{strength}

เรื่องที่ผู้ใช้อยากพัฒนาเพิ่มเติม:
{improvement}

ช่วยตอบแบบเป็นธรรมชาติ เหมือนโค้ชกำลังสะท้อน feedback ให้ผู้ใช้จริง

แนวทางในการตอบ:
- เริ่มจากสะท้อนสิ่งที่ผู้ใช้ได้ลองทำและผลที่เกิดขึ้น
- ชี้ให้เห็นสิ่งที่ผู้ใช้ทำได้ดีจากข้อมูลที่มี
- ต่อด้วยการชวนคิดต่อในเรื่องที่ยังอยากพัฒนา
- ตอนท้ายชวนให้ผู้ใช้ลองสังเกตหรือทดลองต่อในรอบถัดไปอย่างนุ่มนวล

ข้อสำคัญ:
- อย่าตอบแบบแบ่งหัวข้อชัดเกินไป
- อย่าใช้คำชมลอย ๆ
- ให้ลื่นเหมือนคนคุยต่อเนื่องกัน
- ให้ใส่ tag html คำสำคัญหรือ keyword ให้สวยงาม
- ใช้ภาษาที่อบอุ่น เข้าใจง่าย และไม่ยาวเกินไป
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )

        content = response.choices[0].message.content.strip()

        return {
            "ok": True,
            "content": content,
        }

    except Exception as e:
        # fallback แบบไม่เรียก model ซ้ำ
        safe_applied = (applied_action or "").strip()
        safe_result = (result or "").strip()
        safe_strength = (strength or "").strip()
        safe_improvement = (improvement or "").strip()

        parts = []

        first_line = "จากที่คุณเล่ามา"
        if safe_applied and safe_applied != "unknown":
            first_line += f" คุณได้ลอง<strong>{safe_applied}</strong>"
        if safe_result and safe_result != "unknown":
            first_line += f" และเห็นว่า<strong>{safe_result}</strong>"
        first_line += " ซึ่งสะท้อนว่าคุณได้เริ่มนำสิ่งที่ตั้งใจพัฒนาไปใช้จริงแล้ว"
        parts.append(f"<p>{first_line}</p>")

        if safe_strength and safe_strength != "unknown":
            parts.append(
                f"<p>อีกจุดที่น่าสนใจคือคุณมองเห็นว่าตัวเอง<strong>{safe_strength}</strong> "
                f"ตรงนี้เป็นสัญญาณที่ดีมาก เพราะการสังเกตสิ่งที่ทำได้ดีจะช่วยให้ต่อยอดได้ชัดขึ้น</p>"
            )

        if safe_improvement and safe_improvement != "unknown":
            parts.append(
                f"<p>ส่วนเรื่อง<strong>{safe_improvement}</strong>ที่คุณยังอยากพัฒนาต่อ "
                f"อาจลองค่อย ๆ สังเกตในรอบถัดไปก็ได้ครับว่า เมื่อเจอสถานการณ์จริงอีกครั้ง "
                f"คุณอยากลองปรับอะไรเพิ่มอีกนิดเพื่อให้ใกล้กับแบบที่ต้องการมากขึ้น</p>"
            )
        else:
            parts.append(
                "<p>จากตรงนี้ คุณอาจลองสังเกตตัวเองต่ออีกนิดในสถานการณ์ครั้งถัดไปก็ได้ครับ "
                "ว่าอะไรที่ช่วยให้ผลลัพธ์ดีขึ้น และอะไรที่อยากค่อย ๆ ปรับเพิ่มเติม</p>"
            )

        content = "<div>" + "".join(parts) + "</div>"

        return {
            "ok": True,
            "content": content,
        }