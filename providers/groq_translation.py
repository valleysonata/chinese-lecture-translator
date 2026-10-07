import time
from typing import Dict, Any, Optional
from groq import Groq

from providers.base import TranslationEngine
from config import TranslationConfig, DEFAULT_CONFIG

SYSTEM_PROMPT = """You are a strict, real-time simultaneous interpreter for a Computer Science lecture.
Your job is to translate spoken transcript into natural English.

CORE PRINCIPLE: Reconstruct grammar aggressively, reconstruct meaning conservatively.

STRICT RULES:
1. NEVER invent context, extrapolate, or add details not in the input.
2. Short phrases must stay short (e.g. '謝謝大家' -> 'Thank you, everyone.'). Never add YouTube-like outros, welcomes, or conversational filler.
3. If an utterance is an incomplete sentence or trail-off (e.g. '如果這個 node...'), DO NOT finish it. Translate only what was said: 'If this node...'.
4. Smooth out awkward spoken grammar and Chinese-English code-switching into natural English (e.g. '我們就需要做with rotation' -> 'We need to perform a rotation.').
5. Keep all CS technical terminology in English (binary tree, AVL tree, balance factor, node, rotation, pointer, recursion).
6. If the input is already in English, pass it through directly in natural English.
7. Output ONLY the English translation. Never explain, apologize, or add quotation marks.
"""

class GroqTranslationEngine(TranslationEngine):
    """
    Fast English translation using Groq high-speed LLM (Qwen/GPT-OSS).
    Produces low-latency idiomatic English translations preserving CS terminology.
    """
    def __init__(self, api_key: Optional[str] = None, config: Optional[TranslationConfig] = None):
        self.config = config or DEFAULT_CONFIG.translation
        self.api_key = api_key or DEFAULT_CONFIG.groq_api_key

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in your environment or in a .env file."
            )

        self.client = Groq(api_key=self.api_key, timeout=self.config.timeout_seconds)
        self.model = self.config.model

    def translate(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Translates a Mandarin / mixed transcript chunk into natural English.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "success": True,
                "translation": "",
                "latency": 0.0,
                "error": None
            }

        start_time = time.time()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        user_content = f"Transcript to translate: {clean_text}"
        if context:
            user_content = f"[Recent Context: {context}]\n" + user_content

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            latency = time.time() - start_time
            raw_translation = response.choices[0].message.content or ""
            translation = raw_translation.strip().strip('"').strip("'")

            return {
                "success": True,
                "translation": translation,
                "latency": round(latency, 3),
                "error": None
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "translation": "",
                "latency": round(latency, 3),
                "error": str(e)
            }
