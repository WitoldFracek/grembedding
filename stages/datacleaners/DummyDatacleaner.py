from loguru import logger

from stages.datacleaners.DataCleaner import DataCleaner


class DummyDatacleaner(DataCleaner):
    def clean_data(self, dataset: str) -> None:
        train, test = self.load_dataset(dataset)
        train.rename(columns={'text': 'clean_text'}, inplace=True)
        test.rename(columns={'text': 'clean_text'}, inplace=True)
        self.save_dataframe_as_parquet(dataset, train, test)

