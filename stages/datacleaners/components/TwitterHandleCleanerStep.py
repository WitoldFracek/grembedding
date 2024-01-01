import re

from stages.datacleaners.components.GenericRegexCleaner import GenericRegexCleaner


class TwitterHandleCleanerStep(GenericRegexCleaner):
    # USER_HANDLE_REGEX = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)")
    # https://stackoverflow.com/questions/2304632/regex-for-twitter-username

    # https://regex101.com/r/U0lyCv/1
    USER_HANDLE_REGEX = re.compile(
        r"@[\w+ą-ŹóÓ]{1,15}"
    )

    def __init__(self, placeholder: str = "@user"):
        super().__init__(self.USER_HANDLE_REGEX, placeholder=placeholder)
