from stages.dataloaders.DataLoader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetXD(DataLoader):

    def create_dataset(self) -> None:
        
        path = DataLoader.get_file_path("XDData.csv")
        df = pd.read_csv(path, sep=';', header=0, names=['text', 'label'])
        df_train, df_test = train_test_split(df)
        self.save_dataset(df_train, df_test)

