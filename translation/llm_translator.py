import os
import logging
from typing import List, Optional

from translation.base import BaseTranslator, TranslationResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional simultaneous interpreter. Translate the following speech transcription into {target_lang}.

Rules:
- Translate naturally and fluently, not word-by-word
- Preserve technical terms, proper nouns, and acronyms
- Keep the same tone and register as the original
- If context sentences are provided, maintain terminology consistency with them
- Output ONLY the translation, no explanations or notes
- If the input is already in the target language, return it as-is"""

LANG_NAMES = {
    "zh": "Chinese (Simplified)",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
}


class LLMTranslator(BaseTranslator):
    """Translation engine using OpenAI-compatible LLM API."""

    def __init__(self, config):
        self.config = config
        import openai
        self.client = openai.OpenAI(
            api_key=config.api_key or self._get_env_key(),
            base_url=config.api_base_url or "https://api.openai.com/v1",
        )
        self.model = config.model or "gpt-4o-mini"
        logger.info(f"LLM Translator initialized (model={self.model})")

    @staticmethod
    def _get_env_key() -> str:
        return os.environ.get(
            "TRANSLATION_API_KEY",
            os.environ.get("OPENAI_API_KEY", ""),
        )

    def translate(self, text: str, source_lang: str = "en",
                  target_lang: str = "zh",
                  context: Optional[List[str]] = None) -> TranslationResult:
        if not text or not text.strip():
            return TranslationResult(
                source_text=text, translated_text="",
                source_lang=source_lang, target_lang=target_lang,
            )

        target_name = LANG_NAMES.get(target_lang, target_lang)
        system = SYSTEM_PROMPT.format(target_lang=target_name)

        user_parts = []
        if context:
            recent = context[-self.config.context_sentences:]
            ctx_text = "\n".join(f"- {s}" for s in recent)
            user_parts.append(f"Previous translations for context:\n{ctx_text}\n")
        user_parts.append(f"Translate:\n{text}")
        user_content = "\n".join(user_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            translated = response.choices[0].message.content.strip()
            return TranslationResult(
                source_text=text, translated_text=translated,
                source_lang=source_lang, target_lang=target_lang,
            )
        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            return TranslationResult(
                source_text=text, translated_text=f"[Translation Error: {e}]",
                source_lang=source_lang, target_lang=target_lang,
            )
