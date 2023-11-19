from typing import Dict

import sklearn.svm as svm
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from stages.models.Model import Model


class SVC(Model):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
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
        logger.info(f"Fit/transform with scaler complete")

        clf = svm.SVC(**params)

        logger.info(f"Fitting SVC classifier...")
        clf.fit(X_train, y_train)
        logger.info("Predicting with CSV classifier...")
        y_pred = clf.predict(X_test)

        logger.info(f"Params: {params}, acc: {accuracy_score(y_test, y_pred)}")
        metrics = {"accuracy": accuracy_score(y_test, y_pred)}

        self.save_results(
            experiment_name=dataset,
            run_name=f"{datacleaner}-{vectorizer}-{self.__class__.__name__}",
            params=params,
            metrics=metrics,
            clf=clf
        )

        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics)
