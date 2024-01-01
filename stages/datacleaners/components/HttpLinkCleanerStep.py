import re
from typing import Optional

from stages.datacleaners.components.GenericRegexCleaner import GenericRegexCleaner


class HttpLinkCleanerStep(GenericRegexCleaner):
    # src = https://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url
    HTTP_PATTERN = re.compile(
        r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    )

    def __init__(self, placeholder: Optional[str] = None):
        super().__init__(self.HTTP_PATTERN, placeholder=placeholder)
