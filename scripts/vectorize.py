import sys
from objects.vectorizers.Vectorizer import Vectorizer
from objects.vectorizers.CountVectorizer import CountVectorizer

def main():

    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    vectorizer: str = sys.argv[3]

    v = globals()[vectorizer]()
    v.vectorize(dataset, datacleaner)

if __name__ == "__main__":
    main()