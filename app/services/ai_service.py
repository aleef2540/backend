from dotenv import load_dotenv
load_dotenv()

import json
import re



from openai import OpenAI


from app.schemas_all import IntentResult

from app.services.call_ai import call_openai_chat_full, call_openai_chat_stream_full


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

    result = await call_openai_chat_full(
    model="gpt-4.1-mini",
    system_prompt=system_prompt,
    user_prompt=user_message,
    temperature=0.1,
    )

    text = result["content"] or ""
    text = clean_json(text)

    try:
        data = json.loads(text)
        intent = data.get("intent", "general")
    except:
        intent = "general"

    if intent not in ["greeting", "general", "learning"]:
        intent = "general"

    return IntentResult(intent=intent)

async def reply_greeting(user_message: str) -> str:
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

    content = await call_openai_chat_full(
    model=OPENAI_MODEL_ROUTER,
    system_prompt=system_prompt,
    user_prompt=user_message,
    temperature=0.7,
    )

    return content["content"]

async def reply_general(user_message: str) -> str:

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

    content = await call_openai_chat_full(
    model=OPENAI_MODEL_ROUTER,
    system_prompt=system_prompt,
    user_prompt=user_message,
    temperature=0.7,
    )

    return content["content"]