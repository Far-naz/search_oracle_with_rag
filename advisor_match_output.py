from dataclasses import dataclass, field

from advisors_data import Advisor

@dataclass
class MatchAdvisor:
    advisor:Advisor
    score: float
    document: str