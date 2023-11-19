import re

from stages.datacleaners.components.GenericRegexCleaner import GenericRegexCleaner


class TwitterHandleCleanerStep(GenericRegexCleaner):
    USER_HANDLE_REGEX = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)")
    # https://stackoverflow.com/questions/2304632/regex-for-twitter-username

    def __init__(self, placeholder: str = "@user"):
        super().__init__(self.USER_HANDLE_REGEX, placeholder=placeholder)
