from dotenv import load_dotenv
load_dotenv()

import json
import re
from typing import Any


from openai import OpenAI

from app.schemas import ChatIntentOutput

# ปรับตาม env ของคุณเอง
OPENAI_MODEL_ROUTER = "gpt-4.1-mini"
client = OpenAI()


def clean_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text)
    return text.strip()


def safe_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None
    

def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_parse_intent(text: str) -> ChatIntentOutput:
    cleaned = _strip_json_fence(text)

    try:
        data = json.loads(cleaned)
    except Exception:
        # fallback ถ้า parse ไม่ได้
        return ChatIntentOutput(
            intent="general",
            reason="ไม่สามารถ parse JSON จากโมเดลได้ จึง fallback เป็น general",
            topic="unknown",
            learning_need="unknown",
        )

    intent = data.get("intent", "general")
    if intent not in {"greeting", "general", "learning"}:
        intent = "general"

    topic = str(data.get("topic", "unknown") or "unknown").strip()
    learning_need = str(data.get("learning_need", "unknown") or "unknown").strip()
    reason = str(data.get("reason", "") or "").strip()

    if topic == "":
        topic = "unknown"
    if learning_need == "":
        learning_need = "unknown"

    return ChatIntentOutput(
        intent=intent,
        reason=reason,
        topic=topic,
        learning_need=learning_need,
    )


async def detect_intent(user_message: str) -> ChatIntentOutput:
    system_prompt = """คุณคือ AI ที่มีหน้าที่วิเคราะห์ข้อความของผู้ใช้ เพื่อจำแนก intent, topic และ learning_need

ให้ตอบเป็น JSON เท่านั้น และต้องตอบตามกฎนี้อย่างเคร่งครัด

=====================
1) intent
=====================
ให้เลือก intent เพียง 1 ค่าเท่านั้น:
- greeting = ข้อความทักทายสั้น ๆ
- general = พูดคุยทั่วไป ระบายความรู้สึก หรือสนทนาทั่วไปที่ยังไม่มีเป้าหมายพัฒนา/เรียนรู้ชัดเจน
- learning = ผู้ใช้มีปัญหา อยากพัฒนา อยากเรียนรู้ อยากเข้าใจเพิ่มเติม หรือขอคำปรึกษา

=====================
2) topic
=====================
topic คือ “หัวข้อหลักที่ผู้ใช้พูดถึงจริง ๆ” ดดยต้องมีเรื่องชัดเจนถ้าตอบลอยๆเช่นอยากพัฒนาตัวเองถือว่าไม่ได้

กฎ:
- ห้ามเดาหัวข้อเอง
- ถ้าข้อความยังไม่ระบุหัวข้อชัดเจน ให้ตอบ topic = "unknown"

=====================
3) learning_need
=====================
learning_need คือ “สิ่งที่ผู้ใช้อยากรู้หรืออยากเข้าใจเกี่ยวกับ topic”
ให้เลือกเพียง 1 ค่าจากรายการนี้เท่านั้น:
- คืออะไร = อยากรู้ว่าเรื่องนั้นคืออะไร / ความหมายคืออะไร
- อะไรบ้าง = อยากรู้ว่าเรื่องนั้นมีอะไรบ้าง / องค์ประกอบมีอะไรบ้าง
- เป็นอย่างไร = อยากรู้ว่าเรื่องนั้นเป็นอย่างไร / ทำอย่างไร / มีเทคนิคอย่างไร
- ต่างกันอย่างไร = อยากรู้ว่าสิ่ง 2 อย่างต่างกันอย่างไร
- มีอะไรบ้าง = อยากรู้ว่าบทบาทนั้นควรมีคุณสมบัติอะไรบ้าง
- ต้องทำอย่างไร = อยากรู้ว่าควรแก้ปัญหาอย่างไร / ควรปฏิบัติอย่างไร
- unknown = ยังไม่สามารถระบุได้จากข้อความของผู้ใช้

แนวทางตีความ learning_need:
- ถ้าผู้ใช้ถามว่า "...คืออะไร" → คืออะไร
- ถ้าผู้ใช้ถามว่า "...มีอะไรบ้าง" → มีอะไรบ้าง
- ถ้าผู้ใช้ถามว่า "...เป็นอย่างไร" หรือ "ทำอย่างไร" หรือ "มีเทคนิคอะไร" → เป็นอย่างไร
- ถ้าผู้ใช้ถามว่า "...ต่างกันอย่างไร" → ต่างกันอย่างไร
- ถ้าผู้ใช้ถามว่า "บทบาท...ควรมีอะไรบ้าง" → มีอะไรบ้าง
- ถ้าผู้ใช้พูดถึงปัญหา และอยากรู้วิธีแก้ / วิธีปฏิบัติ → ต้องทำอย่างไร
- ถ้าผู้ใช้พอยากพัฒนา / วิธีปฏิบัติ → ต้องทำอย่างไร
- ถ้าผู้ใช้ "ไม่ได้ถาม" หรือ "ไม่ได้ระบุความต้องการชัดเจน"
→ ให้ learning_need = "unknown" เท่านั้น
→ ห้ามตีความเองว่าเป็น "คืออะไร" หรือ "ต้องทำอย่างไร"
- ประโยคที่เป็นเพียงการ "เล่าปัญหา" หรือ "บอกสถานการณ์"
→ ให้ learning_need = "unknown"

=====================
4) กฎการตีความ intent
=====================
- ถ้ามีคำหรือความหมายในลักษณะนี้:
"มีปัญหา", "อยากพัฒนา", "อยากเรียนรู้", "อยากเข้าใจ", "อยากปรึกษา", "ทำไม่ค่อยได้", "ไม่รู้จะทำยังไง", "ควรทำอย่างไร"
→ ให้เป็น learning ทันที
- ถ้าเป็นเพียงข้อความทักทายสั้น ๆ → greeting
- ถ้าเป็นการพูดคุยทั่วไป แต่ยังไม่มีเป้าหมายพัฒนา/เรียนรู้ชัดเจน → general

=====================
5) กฎความสัมพันธ์ของข้อมูล
=====================
- ถ้า intent = greeting หรือ general
→ ให้ topic = "unknown"
→ ให้ learning_need = "unknown"

- ถ้า intent = learning แต่ยังไม่รู้หัวข้อแน่ชัด
→ ให้ topic = "unknown"

- ถ้า intent = learning และรู้ topic แล้ว แต่ยังไม่รู้ว่าผู้ใช้อยากรู้ในมุมไหน
→ ให้ learning_need = "unknown"

- ห้ามสร้าง topic หรือ learning_need เกินจากข้อมูลที่ผู้ใช้ให้มา

=====================
6) reason
=====================
- reason ต้องสั้น ไม่เกิน 1 ประโยค
- อธิบายสั้น ๆ ว่าทำไมจึงจัดอยู่ใน intent นั้น

=====================
7) ตัวอย่าง
=====================

ผู้ใช้: "สวัสดี"
{
"intent": "greeting",
"reason": "เป็นข้อความทักทายสั้น ๆ",
"topic": "unknown",
"learning_need": "unknown"
}

ผู้ใช้: "วันนี้เหนื่อยมากเลย"
{
"intent": "general",
"reason": "เป็นการพูดคุยทั่วไป ยังไม่มีเป้าหมายพัฒนาชัดเจน",
"topic": "unknown",
"learning_need": "unknown"
}

ผู้ใช้: "มีเรื่องอยากปรึกษา"
{
"intent": "learning",
"reason": "ผู้ใช้ต้องการขอคำปรึกษา",
"topic": "unknown",
"learning_need": "unknown"
}

ผู้ใช้: "อยากพัฒนาการสื่อสารกับทีม"
{
"intent": "learning",
"reason": "ผู้ใช้มีเป้าหมายพัฒนาทักษะอย่างชัดเจน",
"topic": "การสื่อสารกับทีม",
"learning_need": "ต้องทำอย่างไร"
}

ผู้ใช้: "การบริหารเวลาคืออะไร"
{
"intent": "learning",
"reason": "ผู้ใช้ต้องการเรียนรู้ความหมายของหัวข้อ",
"topic": "การบริหารเวลา",
"learning_need": "คืออะไร"
}

ผู้ใช้: "ภาวะผู้นำที่ดีมีอะไรบ้าง"
{
"intent": "learning",
"reason": "ผู้ใช้ต้องการรู้องค์ประกอบของหัวข้อ",
"topic": "ภาวะผู้นำ",
"learning_need": "มีอะไรบ้าง"
}

ผู้ใช้: "การโค้ชกับการสอนงานต่างกันอย่างไร"
{
"intent": "learning",
"reason": "ผู้ใช้ต้องการเปรียบเทียบความแตกต่าง",
"topic": "การโค้ชกับการสอนงาน",
"learning_need": "ต่างกันอย่างไร"
}

ผู้ใช้: "มีปัญหาเรื่องการสั่งงานลูกน้องในที่ทำงาน"
{
"intent": "learning",
"reason": "ผู้ใช้กำลังสะท้อนปัญหาที่ต้องการแนวทางแก้ไข",
"topic": "การสั่งงานลูกน้องในที่ทำงาน",
"learning_need": "unknown"
}

ผู้ใช้: "หัวหน้างานที่ดีควรมีอะไรบ้าง"
{
"intent": "learning",
"reason": "ผู้ใช้ต้องการรู้คุณสมบัติของบทบาท",
"topic": "หัวหน้างาน",
"learning_need": "มีอะไรบ้าง"
}

=====================
8) รูปแบบคำตอบ
=====================
ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON

{
"intent": "greeting|general|learning",
"reason": "string",
"topic": "string",
"learning_need": "คืออะไร|อะไรบ้าง|เป็นอย่างไร|ต่างกันอย่างไร|มีอะไรบ้าง|ต้องทำอย่างไร|unknown"
}"""

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
    decoded = safe_parse(text)

    if not isinstance(decoded, dict):
        return ChatIntentOutput(
            intent="general",
            reason="parse_error",
            topic="unknown",
            learning_need="unknown",
        )

    intent = str(decoded.get("intent", "general")).strip()
    reason = str(decoded.get("reason", "")).strip()
    topic = str(decoded.get("topic", "unknown")).strip() or "unknown"
    learning_need = str(decoded.get("learning_need", "unknown")).strip() or "unknown"

    allowed_intents = ["greeting", "general", "learning"]
    if intent not in allowed_intents:
        intent = "general"

    allowed_needs = [
        "คืออะไร",
        "อะไรบ้าง",
        "เป็นอย่างไร",
        "ต่างกันอย่างไร",
        "มีอะไรบ้าง",
        "ต้องทำอย่างไร",
        "unknown",
    ]
    if learning_need not in allowed_needs:
        learning_need = "unknown"

    return ChatIntentOutput(
        intent=intent,
        reason=reason,
        topic=topic if topic else "unknown",
        learning_need=learning_need,
    )


async def reply_greeting(user_message: str) -> str:
    client = OpenAI()

    system_prompt = """
คุณคือ AI coaching ที่พูดคุยกับผู้ใช้อย่างเป็นธรรมชาติ สุภาพ และเป็นมิตร

สถานการณ์: ผู้ใช้เพิ่งทักทาย

หน้าที่:
- ตอบทักทายกลับแบบเป็นธรรมชาติ
- ใช้ภาษาคน ไม่แข็ง ไม่เป็นทางการเกินไป
- ชวนผู้ใช้เล่าต่อ

ข้อห้าม:
- ห้ามตอบยาว
- ห้ามเป็น bullet list
- ห้ามถามหลายคำถามซ้อน

รูปแบบ:
- 1-2 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถามเปิด
""".strip()

    response = client.chat.completions.create(
        model=OPENAI_MODEL_ROUTER,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def reply_general(user_message: str) -> str:
    client = OpenAI()

    system_prompt = """
คุณคือ AI coaching ที่รับฟังผู้ใช้อย่างเป็นธรรมชาติ สุภาพ และเป็นมิตร

สถานการณ์: ผู้ใช้กำลังพูดคุยทั่วไปหรือระบายความรู้สึก แต่ยังไม่มีเป้าหมายการพัฒนาชัดเจน

หน้าที่:
- ตอบอย่างเข้าใจ เป็นธรรมชาติ
- ไม่ยาวเกินไป
- ชวนคุยต่อแบบเปิด
- ยังไม่ต้องรีบสรุปเป็นแผนการเรียนรู้

ข้อห้าม:
- ห้ามตอบเป็น bullet list
- ห้ามแข็งหรือเหมือน chatbot ระบบ
""".strip()

    response = client.chat.completions.create(
        model=OPENAI_MODEL_ROUTER,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()