import time
from typing import Dict, Any, Optional
from groq import Groq

from providers.base import TranslationEngine
from config import TranslationConfig, DEFAULT_CONFIG

SYSTEM_PROMPT = """You are a real-time speech interpreter for an NYCU (National Yang Ming Chiao Tung University) Computer Science lecture.
The professor speaks in Mandarin with English CS terms (code-switching).
Your task is to translate the spoken transcript into clear, natural, idiomatic English.

CRITICAL RULES:
1. Speak natural English: Do NOT translate word-for-word literally (avoid awkward syntax like "we make it right rotate"). Use proper CS English ("perform a right rotation").
2. Preserve meaning and details: Do NOT aggressively summarize. Retain conditions ("if balance factor > 1"), actions, numbers, variables, and algorithm steps.
3. Keep technical CS terms in English: (e.g. binary tree, AVL tree, balance factor, right rotation, pointer, recursion, array, stack, queue, complexity).
4. If the input contains mixed English and Chinese, preserve the English technical terms seamlessly.
5. Output ONLY the final English translation (1 to 2 concise sentences). Do NOT add notes, quotes, or conversational filler.
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
