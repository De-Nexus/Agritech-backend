import os
import json
from groq import Groq
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY", "").strip())


def get_disease_info(disease_name: str) -> dict:
    prompt = f"""
You are an experienced agricultural expert and plant pathologist. A plant disease detection system has identified the following disease: "{disease_name}".

Return ONLY a valid JSON object with exactly this structure, no extra text:
{{
    "causes": ["cause 1", "cause 2", "cause 3"],
    "treatments": ["treatment 1", "treatment 2", "treatment 3"],
    "prevention": ["prevention 1", "prevention 2", "prevention 3"]
}}

Rules:
- Each array must have exactly 3 separate string items
- Each item is one sentence only (15-25 words)
- Do NOT combine multiple points into one string
- Do NOT add any text outside the JSON
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in Groq response")
        text = text[start:end]

        return json.loads(text)

    except Exception as e:
        print(f"Groq API error: {e}")
        raise HTTPException(status_code=502, detail="AI service error")
