import re
from typing import Dict, Tuple, Optional

class PIIAnonymizer:
    """Privacy-preserving entity redaction and reversible de-identification module.
    Safeguards PII (Personal Identifiable Information) before customer ticket text
    is transmitted to external cloud Teacher LLMs (Groq, OpenAI, Anthropic)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # Regex patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
        self.card_pattern = re.compile(r'\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{15,16}\b')
        self.phone_pattern = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}\b')
        self.order_id_pattern = re.compile(r'\b(?:Order\s*#?|INV[-#:\s]*|REF[-#:\s]*|TXN[-#:\s]*|ORD[-#:\s]*|#)\s*([A-Za-z0-9]{4,15})\b', re.IGNORECASE)

    def redact(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Redacts PII from ticket text and returns (redacted_text, restore_mapping)."""
        if not self.enabled or not text:
            return text, {}

        mapping: Dict[str, str] = {}
        counter = {"email": 1, "card": 1, "phone": 1, "order": 1}

        # 1. Redact Emails
        def _replace_email(match):
            placeholder = f"[EMAIL_{counter['email']}]"
            counter["email"] += 1
            val = match.group(0)
            mapping[placeholder] = val
            return placeholder

        redacted = self.email_pattern.sub(_replace_email, text)

        # 2. Redact Credit Cards FIRST (to avoid phone regex overlapping credit card blocks)
        def _replace_card(match):
            placeholder = f"[CARD_{counter['card']}]"
            counter["card"] += 1
            val = match.group(0)
            mapping[placeholder] = val
            return placeholder

        redacted = self.card_pattern.sub(_replace_card, redacted)

        # 3. Redact Order / Transaction IDs
        def _replace_order(match):
            placeholder = f"[ORDER_ID_{counter['order']}]"
            counter["order"] += 1
            val = match.group(0)
            mapping[placeholder] = val
            return placeholder

        redacted = self.order_id_pattern.sub(_replace_order, redacted)

        # 4. Redact Phone Numbers
        def _replace_phone(match):
            val = match.group(0).strip()
            digits = re.sub(r'\D', '', val)
            if len(digits) >= 10:
                placeholder = f"[PHONE_{counter['phone']}]"
                counter["phone"] += 1
                mapping[placeholder] = val
                return placeholder
            return val

        redacted = self.phone_pattern.sub(_replace_phone, redacted)

        return redacted, mapping

    def restore(self, text: str, mapping: Dict[str, str]) -> str:
        """Restores redacted placeholders back to original entity values."""
        if not text or not mapping:
            return text

        restored = text
        for placeholder, original in mapping.items():
            restored = restored.replace(placeholder, original)
        return restored

    def redact_for_logging(self, text: str) -> str:
        """One-way redaction for safe diagnostic application logging."""
        if not text:
            return text
        redacted = self.email_pattern.sub("[EMAIL_REDACTED]", text)
        redacted = self.card_pattern.sub("[CARD_REDACTED]", redacted)
        redacted = self.phone_pattern.sub("[PHONE_REDACTED]", redacted)
        return redacted

# Global singleton
_anonymizer_instance: Optional[PIIAnonymizer] = None

def get_pii_anonymizer() -> PIIAnonymizer:
    global _anonymizer_instance
    if _anonymizer_instance is None:
        _anonymizer_instance = PIIAnonymizer(enabled=True)
    return _anonymizer_instance
