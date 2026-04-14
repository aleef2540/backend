from dotenv import load_dotenv
load_dotenv()

import json
import re
from typing import Any


from openai import OpenAI

from app.schemas import ChatIntentOutput
from app.schemas import IntentResult

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

async def detect_intent(user_message: str) -> IntentResult:

    system_prompt = """คุณคือ AI ที่มีหน้าที่จำแนก intent ของข้อความผู้ใช้

        ให้เลือกเพียง 1 ค่าเท่านั้น:
        - greeting = ข้อความทักทายสั้น ๆ เช่น สวัสดี หวัดดี hello hi
        - general = พูดคุยทั่วไป ระบายความรู้สึก ยังไม่มีเป้าหมายพัฒนา
        - learning = ผู้ใช้มีปัญหา อยากพัฒนา อยากเรียนรู้ หรือขอคำปรึกษา

        กฎ:
        - ถ้ามีคำว่า "อยากพัฒนา", "มีปัญหา", "ไม่รู้จะทำยังไง", "ควรทำอย่างไร" → learning
        - ถ้าเป็นแค่ทักทาย → greeting
        - ที่เหลือ → general

        ตอบเป็น JSON เท่านั้น:

        {
        "intent": "greeting|general|learning"
        }
        """

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

    if intent not in ["greeting", "general", "learning"]:
        intent = "general"

    return IntentResult(intent=intent)

async def call_openai_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# async def detect_intent(user_message: str) -> ChatIntentOutput:
#     system_prompt = """คุณคือ AI ที่มีหน้าที่วิเคราะห์ข้อความของผู้ใช้ เพื่อจำแนก intent, topic และ learning_need

# ให้ตอบเป็น JSON เท่านั้น และต้องตอบตามกฎนี้อย่างเคร่งครัด

# =====================
# 1) intent
# =====================
# ให้เลือก intent เพียง 1 ค่าเท่านั้น:
# - greeting = ข้อความทักทายสั้น ๆ
# - general = พูดคุยทั่วไป ระบายความรู้สึก หรือสนทนาทั่วไปที่ยังไม่มีเป้าหมายพัฒนา/เรียนรู้ชัดเจน
# - learning = ผู้ใช้มีปัญหา อยากพัฒนา อยากเรียนรู้ อยากเข้าใจเพิ่มเติม หรือขอคำปรึกษา

# =====================
# 2) topic
# =====================
# topic คือ “หัวข้อหลักที่ผู้ใช้พูดถึงจริง ๆ” ดดยต้องมีเรื่องชัดเจนถ้าตอบลอยๆเช่นอยากพัฒนาตัวเองถือว่าไม่ได้

# กฎ:
# - ห้ามเดาหัวข้อเอง
# - ถ้าข้อความยังไม่ระบุหัวข้อชัดเจน ให้ตอบ topic = "unknown"

# =====================
# 3) learning_need
# =====================
# learning_need คือ “สิ่งที่ผู้ใช้อยากรู้หรืออยากเข้าใจเกี่ยวกับ topic”
# ให้เลือกเพียง 1 ค่าจากรายการนี้เท่านั้น:
# - คืออะไร = อยากรู้ว่าเรื่องนั้นคืออะไร / ความหมายคืออะไร
# - อะไรบ้าง = อยากรู้ว่าเรื่องนั้นมีอะไรบ้าง / องค์ประกอบมีอะไรบ้าง
# - เป็นอย่างไร = อยากรู้ว่าเรื่องนั้นเป็นอย่างไร / ทำอย่างไร / มีเทคนิคอย่างไร
# - ต่างกันอย่างไร = อยากรู้ว่าสิ่ง 2 อย่างต่างกันอย่างไร
# - มีอะไรบ้าง = อยากรู้ว่าบทบาทนั้นควรมีคุณสมบัติอะไรบ้าง
# - ต้องทำอย่างไร = อยากรู้ว่าควรแก้ปัญหาอย่างไร / ควรปฏิบัติอย่างไร
# - unknown = ยังไม่สามารถระบุได้จากข้อความของผู้ใช้

# แนวทางตีความ learning_need:
# - ถ้าผู้ใช้ถามว่า "...คืออะไร" → คืออะไร
# - ถ้าผู้ใช้ถามว่า "...มีอะไรบ้าง" → มีอะไรบ้าง
# - ถ้าผู้ใช้ถามว่า "...เป็นอย่างไร" หรือ "ทำอย่างไร" หรือ "มีเทคนิคอะไร" → เป็นอย่างไร
# - ถ้าผู้ใช้ถามว่า "...ต่างกันอย่างไร" → ต่างกันอย่างไร
# - ถ้าผู้ใช้ถามว่า "บทบาท...ควรมีอะไรบ้าง" → มีอะไรบ้าง
# - ถ้าผู้ใช้พูดถึงปัญหา และอยากรู้วิธีแก้ / วิธีปฏิบัติ → ต้องทำอย่างไร
# - ถ้าผู้ใช้พอยากพัฒนา / วิธีปฏิบัติ → ต้องทำอย่างไร
# - ถ้าผู้ใช้ "ไม่ได้ถาม" หรือ "ไม่ได้ระบุความต้องการชัดเจน"
# → ให้ learning_need = "unknown" เท่านั้น
# → ห้ามตีความเองว่าเป็น "คืออะไร" หรือ "ต้องทำอย่างไร"
# - ประโยคที่เป็นเพียงการ "เล่าปัญหา" หรือ "บอกสถานการณ์"
# → ให้ learning_need = "unknown"

# =====================
# 4) กฎการตีความ intent
# =====================
# - ถ้ามีคำหรือความหมายในลักษณะนี้:
# "มีปัญหา", "อยากพัฒนา", "อยากเรียนรู้", "อยากเข้าใจ", "อยากปรึกษา", "ทำไม่ค่อยได้", "ไม่รู้จะทำยังไง", "ควรทำอย่างไร"
# → ให้เป็น learning ทันที
# - ถ้าเป็นเพียงข้อความทักทายสั้น ๆ → greeting
# - ถ้าเป็นการพูดคุยทั่วไป แต่ยังไม่มีเป้าหมายพัฒนา/เรียนรู้ชัดเจน → general

# =====================
# 5) กฎความสัมพันธ์ของข้อมูล
# =====================
# - ถ้า intent = greeting หรือ general
# → ให้ topic = "unknown"
# → ให้ learning_need = "unknown"

# - ถ้า intent = learning แต่ยังไม่รู้หัวข้อแน่ชัด
# → ให้ topic = "unknown"

# - ถ้า intent = learning และรู้ topic แล้ว แต่ยังไม่รู้ว่าผู้ใช้อยากรู้ในมุมไหน
# → ให้ learning_need = "unknown"

# - ห้ามสร้าง topic หรือ learning_need เกินจากข้อมูลที่ผู้ใช้ให้มา

# =====================
# 6) reason
# =====================
# - reason ต้องสั้น ไม่เกิน 1 ประโยค
# - อธิบายสั้น ๆ ว่าทำไมจึงจัดอยู่ใน intent นั้น

# =====================
# 7) ตัวอย่าง
# =====================

# ผู้ใช้: "สวัสดี"
# {
# "intent": "greeting",
# "reason": "เป็นข้อความทักทายสั้น ๆ",
# "topic": "unknown",
# "learning_need": "unknown"
# }

# ผู้ใช้: "วันนี้เหนื่อยมากเลย"
# {
# "intent": "general",
# "reason": "เป็นการพูดคุยทั่วไป ยังไม่มีเป้าหมายพัฒนาชัดเจน",
# "topic": "unknown",
# "learning_need": "unknown"
# }

# ผู้ใช้: "มีเรื่องอยากปรึกษา"
# {
# "intent": "learning",
# "reason": "ผู้ใช้ต้องการขอคำปรึกษา",
# "topic": "unknown",
# "learning_need": "unknown"
# }

# ผู้ใช้: "อยากพัฒนาการสื่อสารกับทีม"
# {
# "intent": "learning",
# "reason": "ผู้ใช้มีเป้าหมายพัฒนาทักษะอย่างชัดเจน",
# "topic": "การสื่อสารกับทีม",
# "learning_need": "ต้องทำอย่างไร"
# }

# ผู้ใช้: "การบริหารเวลาคืออะไร"
# {
# "intent": "learning",
# "reason": "ผู้ใช้ต้องการเรียนรู้ความหมายของหัวข้อ",
# "topic": "การบริหารเวลา",
# "learning_need": "คืออะไร"
# }

# ผู้ใช้: "ภาวะผู้นำที่ดีมีอะไรบ้าง"
# {
# "intent": "learning",
# "reason": "ผู้ใช้ต้องการรู้องค์ประกอบของหัวข้อ",
# "topic": "ภาวะผู้นำ",
# "learning_need": "มีอะไรบ้าง"
# }

# ผู้ใช้: "การโค้ชกับการสอนงานต่างกันอย่างไร"
# {
# "intent": "learning",
# "reason": "ผู้ใช้ต้องการเปรียบเทียบความแตกต่าง",
# "topic": "การโค้ชกับการสอนงาน",
# "learning_need": "ต่างกันอย่างไร"
# }

# ผู้ใช้: "มีปัญหาเรื่องการสั่งงานลูกน้องในที่ทำงาน"
# {
# "intent": "learning",
# "reason": "ผู้ใช้กำลังสะท้อนปัญหาที่ต้องการแนวทางแก้ไข",
# "topic": "การสั่งงานลูกน้องในที่ทำงาน",
# "learning_need": "unknown"
# }

# ผู้ใช้: "หัวหน้างานที่ดีควรมีอะไรบ้าง"
# {
# "intent": "learning",
# "reason": "ผู้ใช้ต้องการรู้คุณสมบัติของบทบาท",
# "topic": "หัวหน้างาน",
# "learning_need": "มีอะไรบ้าง"
# }

# =====================
# 8) รูปแบบคำตอบ
# =====================
# ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON

# {
# "intent": "greeting|general|learning",
# "reason": "string",
# "topic": "string",
# "learning_need": "คืออะไร|อะไรบ้าง|เป็นอย่างไร|ต่างกันอย่างไร|มีอะไรบ้าง|ต้องทำอย่างไร|unknown"
# }"""

#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         temperature=0.1,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message},
#         ],
#     )

#     text = response.choices[0].message.content or ""
#     text = clean_json(text)
#     decoded = safe_parse(text)

#     if not isinstance(decoded, dict):
#         return ChatIntentOutput(
#             intent="general",
#             reason="parse_error",
#             topic="unknown",
#             learning_need="unknown",
#         )

#     intent = str(decoded.get("intent", "general")).strip()
#     reason = str(decoded.get("reason", "")).strip()
#     topic = str(decoded.get("topic", "unknown")).strip() or "unknown"
#     learning_need = str(decoded.get("learning_need", "unknown")).strip() or "unknown"

#     allowed_intents = ["greeting", "general", "learning"]
#     if intent not in allowed_intents:
#         intent = "general"

#     allowed_needs = [
#         "คืออะไร",
#         "อะไรบ้าง",
#         "เป็นอย่างไร",
#         "ต่างกันอย่างไร",
#         "มีอะไรบ้าง",
#         "ต้องทำอย่างไร",
#         "unknown",
#     ]
#     if learning_need not in allowed_needs:
#         learning_need = "unknown"

#     return ChatIntentOutput(
#         intent=intent,
#         reason=reason,
#         topic=topic if topic else "unknown",
#         learning_need=learning_need,
#     )


async def reply_greeting(user_message: str) -> str:
    client = OpenAI()

    system_prompt = """
คุณคือ AI coaching ที่พูดคุยกับผู้ใช้อย่างเป็นธรรมชาติ สุภาพ และเป็นมิตร

            สถานการณ์: ผู้ใช้เพิ่งทักทาย (greeting)

            หน้าที่:
            - ตอบทักทายกลับแบบเป็นธรรมชาติ
            - ใช้ภาษาคน ไม่แข็ง ไม่เป็นทางการเกินไป
            - ชวนผู้ใช้เล่าต่อ

            ข้อห้าม:
            - ห้ามตอบยาว
            - ห้ามเป็น bullet list
            - ห้ามถามหลายคำถามซ้อน

            รูปแบบ:
            - 1–2 ประโยค
            - ประโยคสุดท้ายต้องเป็นคำถามเปิด

            แนวทาง:
            - ชวนไปสู่: ปรึกษา / คุย / เรียนรู้ / พัฒนา
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
คุณคือ AI Coaching Assistant ที่พูดคุยอย่างเป็นธรรมชาติ สุภาพ เป็นมิตร และไม่แข็งเหมือนบอท

            สถานการณ์:
            ผู้ใช้กำลังพูดคุยทั่วไป (general) หรือพูดเรื่องกว้าง ๆ ที่ยังไม่เกี่ยวกับการพัฒนาโดยตรง

            เป้าหมาย:
            - ตอบรับสิ่งที่ผู้ใช้พูดอย่างสั้นและเป็นธรรมชาติ
            - จากนั้นค่อย ๆ พาผู้ใช้เข้าสู่การปรึกษา การเรียนรู้ หรือการพัฒนา
            - ต้องไม่ตีความว่าผู้ใช้อยากได้คำแนะนำในเรื่องที่เพิ่งเล่ามา เว้นแต่ผู้ใช้จะถามตรง ๆ

            หน้าที่:
            - รับข้อความของผู้ใช้แบบสั้น ๆ ก่อน
            - เชื่อมเข้าสู่คำถามปลายเปิดที่ชวนให้ผู้ใช้เล่าต่อ
            - ชวนไปสู่เรื่องที่ผู้ใช้อยากปรึกษา อยากเรียนรู้ หรืออยากพัฒนา

            ข้อห้าม:
            - ห้ามคุยต่อ small talk ในเรื่องเดิม
            - ห้ามถามต่อเฉพาะเรื่องเดิม
            - ห้ามเสนอคำแนะนำจากเรื่องเดิมเอง
            - ห้ามเดาว่าผู้ใช้อยากเรียนรู้เรื่องที่เพิ่งพูดถึง
            - ห้ามใช้ประโยคซ้ำเดิมทุกครั้ง
            - ห้ามตอบยาว
            - ห้ามถามหลายคำถามซ้อน
            - ห้ามใช้โทนแข็ง ทื่อ หรือเหมือน template

            กฎสำคัญ:
            - ถ้าผู้ใช้พูดเรื่องทั่วไป เช่น สัตว์เลี้ยง อากาศ อาหาร เรื่องรอบตัว
            ให้ตอบรับสั้น ๆ เท่านั้น
            - จากนั้นต้อง redirect ไปถามแบบกว้าง ๆ ว่า
            ผู้ใช้มีเรื่องอะไรที่อยากปรึกษา อยากเรียนรู้ หรืออยากพัฒนาหรือไม่
            - ห้ามเปลี่ยนเรื่องทั่วไปนั้นให้กลายเป็นหัวข้อให้คำปรึกษาเอง

            รูปแบบ:
            - ตอบ 1-2 ประโยค
            - ประโยคแรกอาจรับหรือเชื่อมจากสิ่งที่ผู้ใช้พูด
            - ประโยคสุดท้ายต้องเป็นคำถามเปิด เพื่อพาเข้าสู่การคุยต่อเชิงปรึกษา/เรียนรู้/พัฒนา

            ตัวอย่างแนวที่ดี:
            - ยินดีด้วยนะครับ ฟังแล้วเป็นช่วงเวลาที่น่ารักมากเลย ถ้ามีเรื่องไหนที่อยากปรึกษาหรืออยากคุยต่อ ผมยินดีช่วยนะครับ
            - ฟังดูเป็นเรื่องดีเลยครับ ถ้าช่วงนี้มีอะไรที่อยากเรียนรู้หรืออยากพัฒนาเพิ่มเติม ลองเล่าให้ผมฟังได้เลยครับ

            ตัวอย่างแนวที่ไม่ควรใช้:
            - อยากได้คำแนะนำเรื่องการดูแลลูกหมาหรือแม่หมาไหมครับ
            - หมาคลอดกี่ตัวครับ
            - ช่วงนี้ให้นมดีไหมครับ
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