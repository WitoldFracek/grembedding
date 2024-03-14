from stages.vectorizers.components.BertLikeTunableModel import BertLikeTunableModel


class RoBERTaFrozen(BertLikeTunableModel):
    MODEL_PATH = "sdadas/polish-roberta-base-v2"
    TOKENIZER_PATH = "sdadas/polish-roberta-base-v2"

    def __init__(self):
        super().__init__(
            model_path=self.MODEL_PATH,
            tokenizer_path=self.TOKENIZER_PATH,
            max_tokens_num=512,
            is_frozen=True
        )
