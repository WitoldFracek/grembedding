from stages.vectorizers.components.CountVectorizer import CountVectorizer


class CountVectorizer1000(CountVectorizer):

    def __init__(self):
        super().__init__(1000)
