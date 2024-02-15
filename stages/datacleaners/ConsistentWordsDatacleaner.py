from stages.datacleaners.DataCleaner import DataCleaner


class ConsistentWordsDatacleaner(DataCleaner):

    """
    If some escape characters are attached to words like \n, \n\n or \r - separate them
    """

    # TODO - dużo książek ma jakieś zlepione znaki \n \r i inne - warto to chyba oddzielać od słów / albo wgl usuwać
    def clean_data(self, dataset: str) -> None:
        train_df, test_df = self.load_dataset(dataset)

