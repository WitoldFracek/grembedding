from stages.models.Model import Model
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from typing import Dict
from loguru import logger

class SVC(Model):
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params: Dict[str, int | float | str]) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :vectorizer: name of vectorizer that was used to vectorize the data
        :params: params for model
        """
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        clf = svm.SVC(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        logger.info(f"Params: {params}, acc: {accuracy_score(y_test, y_pred)}")
        self.save_results(
            experiment_name = dataset,
            run_name = f"{datacleaner}-{vectorizer}-{self.__class__.__name__}",
            params = params,
            metrics = {"accuracy": accuracy_score(y_test, y_pred)},
            clf = clf
        )
    