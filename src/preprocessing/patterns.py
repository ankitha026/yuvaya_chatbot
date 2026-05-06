# FILE 1 — `src/preprocessing/patterns.py`

"""
patterns.py – Conversational patterns and static responses.
"""

from __future__ import annotations

GREETING_PATTERNS = [
    r"\bhi+\b",
    r"\bhello+\b",
    r"\bhey+\b",
    r"\bhii+\b",
    r"\bheyy+\b",
    r"\bhola\b",
    r"\bnamaste\b",
    r"\byo\b",
    r"\bsup\b",
    r"\bhiya\b",
    r"\bgood morning\b",
    r"\bgood afternoon\b",
    r"\bgood evening\b",
    r"\bwhat'?s up\b",
    r"\bmorning\b",
]

GREETING_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi! What can I help you find?",
    "Hey! How may I assist you today?",
]

THANKS_PATTERNS = [
    r"\bthanks+\b",
    r"\bthank you\b",
    r"\bthankyou\b",
    r"\bthanks a lot\b",
    r"\bmany thanks\b",
    r"\bthx\b",
    r"\bty\b",
]

THANKS_RESPONSES = [
    "You're welcome!",
    "Happy to help!",
    "Glad I could help.",
]

GOODBYE_PATTERNS = [
    r"\bbye+\b",
    r"\bgoodbye\b",
    r"\bsee you\b",
    r"\bsee ya\b",
    r"\btake care\b",
    r"\bcatch you later\b",
]

GOODBYE_RESPONSES = [
    "Goodbye! Take care.",
    "See you again soon!",
    "Have a great day!",
]

ACKNOWLEDGEMENT_PATTERNS = [
    r"\bok\b",
    r"\bokay\b",
    r"\bcool\b",
    r"\bgreat\b",
    r"\bnice\b",
    r"\bgot it\b",
    r"\bunderstood\b",
    r"\balright\b",
]

ACKNOWLEDGEMENT_RESPONSES = [
    "Understood.",
    "Great!",
    "Glad that helped.",
]
