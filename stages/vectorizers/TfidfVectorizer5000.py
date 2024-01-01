from stages.vectorizers.components.TfidfVectorizer import TfidfVectorizer


class TfidfVectorizer5000(TfidfVectorizer):

    def __init__(self):
        params = {
            "max_features": 5000
        }

        super().__init__(params)
