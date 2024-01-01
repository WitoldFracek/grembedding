import re
from typing import Optional

import pandas as pd

from stages.datacleaners.components.base import DataProcessingStep


class GenericRegexCleaner(DataProcessingStep):

    def __init__(self, regex: re.Pattern[str], placeholder: Optional[str] = None):
        self.regex = regex
        self.placeholder = placeholder if placeholder is not None else r""

    def do_transform_inplace(self, df: pd.DataFrame) -> None:
        df["text"] = df["text"].apply(self._run_substitution)

    def _run_substitution(self, text: str) -> str:
        return re.sub(self.regex, self.placeholder, text)
