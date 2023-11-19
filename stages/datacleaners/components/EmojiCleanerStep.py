import re
from typing import Optional

from stages.datacleaners.components.GenericRegexCleaner import GenericRegexCleaner


class EmojiCleanerStep(GenericRegexCleaner):
    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])", flags=re.UNICODE)

    # src = https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1

    def __init__(self, placeholder: Optional[str] = None):
        super().__init__(self.EMOJI_PATTERN, placeholder=placeholder)
