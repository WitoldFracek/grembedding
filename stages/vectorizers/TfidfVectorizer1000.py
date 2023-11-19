from stages.vectorizers.components.TfidfVectorizer import TfidfVectorizer


class TfidfVectorizer1000(TfidfVectorizer):

    def __init__(self):
        params = {
            "max_features": 1000
        }

        super().__init__(params)
