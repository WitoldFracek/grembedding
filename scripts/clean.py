import sys
from objects.datacleaners.DataCleaner import DataCleaner
from objects.datacleaners.LemmatizerSM import LemmatizerSM


def main():

    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]

    dc = globals()[datacleaner]()
    dc.clean_data(dataset)

if __name__ == "__main__":
    main()