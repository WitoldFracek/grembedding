from scripts import utils
import sys
from objects.datacleaners import DataCleaner, LemmatizerSM


def main():

    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    print(dataset, datacleaner)

    sys.path.append(utils._get_main_dir_path())

    print(globals())
    x = globals()[datacleaner]
    print(x)

if __name__ == "__main__":
    main()