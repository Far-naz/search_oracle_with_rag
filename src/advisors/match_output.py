from dataclasses import dataclass

from src.advisors.models import Advisor


@dataclass
class MatchAdvisor:
    advisor: Advisor
    score: float
    document: str
