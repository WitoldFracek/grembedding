import functools

import mlflow


def mlflow_context(func):
    """Designed to wrap Model::evaluate, starts and stops MLFlow run, populates experiment_name/id, run_name/id"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO shady
        assert len(args) == 6, "Passed args do not match Model::evaluate"
        model_instance, dataset, datacleaner, vectorizer, params_name, params = args

        class_name: str = model_instance.__class__.__name__
        run_name: str = f"{datacleaner}-{vectorizer}-{class_name}"
        experiment_name: str = dataset

        default_tags: dict[str, str] = {
            "dataset": dataset,
            "vectorizer": vectorizer,
            "datacleaner": datacleaner,
        }

        mlflow.set_experiment(experiment_name=experiment_name)

        with mlflow.start_run(run_name=run_name,tags=default_tags):
            result = func(model_instance, dataset, datacleaner, vectorizer, params_name, params)
            return result

    return wrapper
