from stages.vectorizers.components.BertLikeTunableModel import BertLikeTunableModel


class HerbertFT(BertLikeTunableModel):
    TOKENIZER_PATH: str = "allegro/herbert-base-cased"
    MODEL_PATH: str = "allegro/herbert-base-cased"

    def __init__(self):
        super().__init__(
            model_path=self.MODEL_PATH,
            tokenizer_path=self.TOKENIZER_PATH,
            max_tokens_num=512
        )
