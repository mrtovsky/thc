from typing import Dict


REPOSITORY_NAME: str = "thc"

DISTILBERT_TOKENS_MAX_LENGTH: int = 512

CLASS_WEIGHTS: Dict[int, float] = {0: 0.3642, 1: 13.2354, 2: 5.5911}
