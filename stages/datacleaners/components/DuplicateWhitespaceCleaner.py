import re
from typing import Optional

from stages.datacleaners.components.GenericRegexCleaner import GenericRegexCleaner


class DuplicateWhitespaceCleaner(GenericRegexCleaner):
    SINGLE_SPACE: str = " "

    REGEX = re.compile(
        r"\s+"
    )

    def __init__(self, placeholder: Optional[str] = SINGLE_SPACE):
        super().__init__(self.REGEX, placeholder=placeholder)
