from dataclasses import dataclass

from advisors_data import Advisor

@dataclass
class MatchAdvisor:
    advisor:Advisor
    score: float
    document: str