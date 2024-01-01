import re
from typing import Optional

from stages.datacleaners.components.GenericRegexCleaner import GenericRegexCleaner


class HtmlTagCleanerStep(GenericRegexCleaner):
    HTML_TAG_REGEX = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    def __init__(self, placeholder: Optional[str] = None):
        super().__init__(self.HTML_TAG_REGEX, placeholder=placeholder)