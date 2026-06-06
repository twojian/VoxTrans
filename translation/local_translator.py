import logging
from typing import List, Optional

from translation.base import BaseTranslator, TranslationResult

logger = logging.getLogger(__name__)


class LocalTranslator(BaseTranslator):
    """Offline translation engine using HuggingFace MarianMT models."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        model_name = self.config.local_model_name or "Helsinki-NLP/opus-mt-en-zh"
        logger.info(f"Loading local translation model: {model_name}")
        try:
            from transformers import MarianMTModel, MarianTokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            logger.info("Local translation model loaded")
        except ImportError:
            logger.error(
                "transformers not installed. "
                "Install with: pip install transformers sentencepiece"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def translate(self, text: str, source_lang: str = "en",
                  target_lang: str = "zh",
                  context: Optional[List[str]] = None) -> TranslationResult:
        if not text or not text.strip():
            return TranslationResult(
                source_text=text, translated_text="",
                source_lang=source_lang, target_lang=target_lang,
            )

        if self.model is None or self.tokenizer is None:
            return TranslationResult(
                source_text=text, translated_text="[Model not loaded]",
                source_lang=source_lang, target_lang=target_lang,
            )

        try:
            inputs = self.tokenizer(text, return_tensors="pt",
                                    padding=True, truncation=True,
                                    max_length=512)
            outputs = self.model.generate(**inputs, max_length=512)
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return TranslationResult(
                source_text=text, translated_text=translated,
                source_lang=source_lang, target_lang=target_lang,
            )
        except Exception as e:
            logger.error(f"Local translation error: {e}")
            return TranslationResult(
                source_text=text, translated_text=f"[Translation Error: {e}]",
                source_lang=source_lang, target_lang=target_lang,
            )

    def close(self):
        self.model = None
        self.tokenizer = None
