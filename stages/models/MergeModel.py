from stages.models.Model import Model
from typing import Dict

### DO NOT USE TO CLASSIFY
### ONLY FOR MERGE PURPOSE
class MergeModel(Model):
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        pass