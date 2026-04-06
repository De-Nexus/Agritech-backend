import os
import json
import base64
from groq import Groq
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY", "").strip())

SUPPORTED_CROPS = ["Corn", "Pepper Bell", "Potato", "Tomato"]


def get_disease_info(disease_name: str, image_bytes: bytes) -> dict:
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = f"""You are an experienced agricultural expert and plant pathologist.

A plant disease detection model has analysed the attached leaf image and predicted: "{disease_name}".

The model only supports these crops: {", ".join(SUPPORTED_CROPS)}.

First, look at the image and decide: is this actually a leaf from one of the supported crops ({", ".join(SUPPORTED_CROPS)})?

Then return ONLY a valid JSON object. No explanation, no markdown, just the raw JSON.

If the leaf IS from a supported crop:
{{
    "supported": true,
    "causes": ["first cause as one sentence", "second cause as one sentence", "third cause as one sentence"],
    "treatments": ["first treatment as one sentence", "second treatment as one sentence", "third treatment as one sentence"],
    "prevention": ["first prevention tip as one sentence", "second prevention tip as one sentence", "third prevention tip as one sentence"]
}}

If the leaf is NOT from a supported crop:
{{
    "supported": false,
    "causes": [],
    "treatments": [],
    "prevention": []
}}

Rules:
- Each array must have exactly 3 separate string items
- Each item is one sentence only (15-25 words)
- Do NOT combine multiple points into one string
- Do NOT add any text outside the JSON"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
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
