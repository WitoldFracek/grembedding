from stages.vectorizers.components.CountVectorizer import CountVectorizer


class CountVectorizer5000(CountVectorizer):

    def __init__(self):
        super().__init__(5000)
