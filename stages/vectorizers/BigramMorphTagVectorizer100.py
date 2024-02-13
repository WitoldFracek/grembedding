from stages.vectorizers.components.BigramMorphTagVectorizer import BigramMorphTagVectorizer


class BigramMorphTagVectorizer100(BigramMorphTagVectorizer):

    def __init__(self):
        super().__init__(100)
