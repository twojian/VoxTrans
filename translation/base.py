from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str = ""
    target_lang: str = "zh"


class BaseTranslator(ABC):
    """Abstract base class for translation engines."""

    @abstractmethod
    def translate(self, text: str, source_lang: str = "en",
                  target_lang: str = "zh",
                  context: Optional[List[str]] = None) -> TranslationResult:
        ...

    def close(self):
        pass


def create_translator(config) -> BaseTranslator:
    engine = config.engine
    if engine == "llm-api":
        from translation.llm_translator import LLMTranslator
        return LLMTranslator(config)
    elif engine == "local":
        from translation.local_translator import LocalTranslator
        return LocalTranslator(config)
    else:
        raise ValueError(f"Unknown translation engine: {engine}")
