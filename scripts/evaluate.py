import sys
from objects.models.Model import Model
from objects.models.SVC import SVC


def main():

    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    vectorizer: str = sys.argv[3]
    model: str = sys.argv[4]

    md = globals()[model]()
    md.evaluate(dataset, datacleaner, vectorizer)

if __name__ == "__main__":
    main() 