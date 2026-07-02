from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np

from .config import (
    _EMAIL_RE,
    _WEB_RE,
    _PHONE_DIGITS_RE,
    _SOCIAL_PREFIX_RE,
    _AT_HANDLE_RE,
    _POSTAL_SHAPE_RE,
    _ADDR_SEPARATOR_RE,
)


@dataclass
class OCRToken:
    text: str
    confidence: float
    bbox: np.ndarray
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def x_min(self) -> float:
        return float(np.min(self.bbox[:, 0]))

    @property
    def x_max(self) -> float:
        return float(np.max(self.bbox[:, 0]))

    @property
    def y_min(self) -> float:
        return float(np.min(self.bbox[:, 1]))

    @property
    def y_max(self) -> float:
        return float(np.max(self.bbox[:, 1]))

    @property
    def char_width_est(self) -> float:
        alpha = sum(c.isalpha() for c in self.text)
        return self.width / alpha if alpha > 0 else self.width / max(len(self.text), 1)


@dataclass
class TextRow:
    tokens: List[OCRToken]
    text: str
    y_center: float
    x_min: float
    x_max: float
    median_height: float
    avg_confidence: float
    layout_role: str = "unknown"
    role_score: float = 0.0
    tps: float = 1.0
    position_frac: float = 0.5

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def has_digit(self) -> bool:
        return any(c.isdigit() for c in self.text)

    @property
    def digit_density(self) -> float:
        return sum(c.isdigit() for c in self.text) / max(len(self.text), 1)

    @property
    def punct_density(self) -> float:
        return sum(c in ",./-#@:;" for c in self.text) / max(len(self.text), 1)

    @property
    def alpha_density(self) -> float:
        return sum(c.isalpha() for c in self.text) / max(len(self.text), 1)

    @property
    def is_all_caps(self) -> bool:
        alpha = [c for c in self.text if c.isalpha()]
        return len(alpha) > 0 and all(c.isupper() for c in alpha)

    @property
    def is_title_case(self) -> bool:
        words = self.text.split()
        return (
            len(words) >= 1
            and sum(w[0].isupper() for w in words if w) / len(words) >= 0.6
        )

    @property
    def has_email_pattern(self) -> bool:
        return bool(_EMAIL_RE.search(self.text))

    @property
    def has_phone_pattern(self) -> bool:
        m = _PHONE_DIGITS_RE.search(self.text)
        if not m:
            return False
        return 7 <= len(__import__("re").sub(r"\D", "", m.group(0))) <= 15

    @property
    def has_web_pattern(self) -> bool:
        return bool(_WEB_RE.search(self.text))

    @property
    def has_postal_pattern(self) -> bool:
        return bool(_POSTAL_SHAPE_RE.search(self.text))

    @property
    def has_social_pattern(self) -> bool:
        return bool(
            _SOCIAL_PREFIX_RE.search(self.text) or _AT_HANDLE_RE.search(self.text)
        )

    @property
    def separator_count(self) -> int:
        return len(_ADDR_SEPARATOR_RE.findall(self.text))


@dataclass
class ContactCard:
    name: str = "—"
    job_title: str = "—"
    company: str = "—"
    email: str = "—"
    phone_1: str = "—"
    phone_2: str = "—"
    phone_3: str = "—"
    website: str = "—"
    address: str = "—"
    linkedin: str = "—"
    twitter: str = "—"
    instagram: str = "—"
    github: str = "—"
    quality_score: str = "🔴 RED"
    raw_text: str = ""
    confidence_avg: float = 0.0
    source_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_vcard(self) -> str:
        lines = ["BEGIN:VCARD", "VERSION:3.0"]
        if self.name != "—":
            lines.append(f"FN:{self.name}")
            p = self.name.split(maxsplit=1)
            lines.append(
                f"N:{p[-1]};{p[0]};;;" if len(p) == 2 else f"N:{self.name};;;;"
            )
        if self.company != "—":
            lines.append(f"ORG:{self.company}")
        if self.job_title != "—":
            lines.append(f"TITLE:{self.job_title}")
        if self.email != "—":
            lines.append(f"EMAIL;TYPE=INTERNET:{self.email}")
        if self.phone_1 != "—":
            lines.append(f"TEL;TYPE=VOICE,PREF:{self.phone_1}")
        if self.phone_2 != "—":
            lines.append(f"TEL;TYPE=VOICE:{self.phone_2}")
        if self.website != "—":
            lines.append(f"URL:{self.website}")
        if self.address != "—":
            lines.append(f"ADR;TYPE=WORK:;;{self.address.replace(', ', '; ')};;;;")
        lines.append("END:VCARD")
        return "\n".join(lines)
