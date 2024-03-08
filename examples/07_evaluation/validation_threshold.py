'''Basic Linear Regression, regularized with ElasticNet.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

Modified following https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html

Usage:

    conda activate <env_name>
    python ./<filename.py>

'''
import sys
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path
import os
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import sklearn
import joblib
import cloudpickle
from mlflow.models import make_metric
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("../data/red-wine-quality.csv")
    # Data artifacts
    train, test = train_test_split(data)
    data_dir = 'data'
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir + '/dataset.csv')
    train.to_csv(data_dir + '/dataset_train.csv')
    test.to_csv(data_dir + '/dataset_test.csv')

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="experiment_model_evaluation")
    #get_exp = mlflow.get_experiment(exp_id)

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run()
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False
    )
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_params({
        "alpha": 0.4,
        "l1_ratio": 0.4
    })

    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    })

    # Baseline model
    baseline_model = DummyRegressor()
    baseline_model.fit(train_x, train_y)

    baseline_predicted_qualites = baseline_model.predict(test_x)

    bl_rmse, bl_mae, bl_r2 = eval_metrics(test_y, baseline_predicted_qualites)

    print("Baseline Dummy model")
    print("  Baseline RMSE: %s" % bl_rmse)
    print("  Baseline MAE: %s" % bl_mae)
    print("  Baseline R2: %s" % bl_r2)

    mlflow.log_metrics({
        "Baseline rmse": bl_rmse,
        "Baseline r2": bl_rmse,
        "Baseline mae": bl_rmse
    })

    # Model artifact: we serialize the model with joblib
    model_dir = 'models'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = model_dir + "/model.pkl"
    joblib.dump(lr, model_path)

    # Artifacts' paths: model and data
    # This dictionary is fetsched later by the mlflow context
    artifacts = {
        "model": model_path,
        "data": data_dir
    }

    # Save baseline model and an artifacts dictionary
    baseline_model_path = model_dir + "/baseline_model.pkl"
    joblib.dump(baseline_model, baseline_model_path)
    baseline_artifacts = {
        "baseline_model": baseline_model_path
    }

    # We create a wrapper class, i.e.,
    # a custom mlflow.pyfunc.PythonModel
    #   https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel
    # We need to define at least two functions:
    # - load_context
    # - predict
    # We can also define further custom functions if we want
    class ModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, artifacts_name):
            # We use the artifacts_name in order to handle both the baseline & the custom model
            self.artifacts_name = artifacts_name

        def load_context(self, context):
            self.model = joblib.load(context.artifacts[self.artifacts_name])

        def predict(self, context, model_input):
            return self.model.predict(model_input.values)

    # Conda environment
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            f"python={sys.version}", # Python version
            "pip",
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                ],
            },
        ],
        "name": "my_env",
    }

    # Log model with all the structures defined above
    # We'll see all the artifacts in the UI: data, models, code, etc.
    mlflow.pyfunc.log_model(
        artifact_path="custom_mlflow_pyfunc", # the path directory which will contain the model
        python_model=ModelWrapper("model"), # a mlflow.pyfunc.PythonModel, defined above
        artifacts=artifacts, # dictionary defined above
        code_path=[str(__file__)], # Code file(s), must be in local dir: "model_customization.py"
        conda_env=conda_env
    )
    
    # Baseline model
    mlflow.pyfunc.log_model(
        artifact_path="baseline_mlflow_pyfunc", # the path directory which will contain the model
        python_model=ModelWrapper("baseline_model"), # a mlflow.pyfunc.PythonModel, defined above
        artifacts=baseline_artifacts, # dictionary defined above
        code_path=[str(__file__)], # Code file(s), must be in local dir: "model_customization.py"
        conda_env=conda_env
    )

    def squared_diff_plus_one(eval_df, _builtin_metrics):
        return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)

    def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
        return builtin_metrics["sum_on_target"] / 2

    squared_diff_plus_one_metric = make_metric(
        eval_fn=squared_diff_plus_one,
        greater_is_better=False,
        name="squared diff plus one"
    )

    sum_on_target_divided_by_two_metric = make_metric(
        eval_fn=sum_on_target_divided_by_two,
        greater_is_better=True,
        name="sum on target divided by two"
    )

    def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
        plt.scatter(eval_df["prediction"], eval_df["target"])
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.title("Targets vs. Predictions")
        plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
        plt.savefig(plot_path)
        return {"example_scatter_plot_artifact": plot_path}

    model_artifact_uri = mlflow.get_artifact_uri("custom_mlflow_pyfunc")

    # After training and logging both the baseline and the (custom) model
    # We define a thresholds dictionary which 
    thresholds = {
        "mean_squared_error": MetricThreshold(
            threshold=0.6,  # Maximum MSE threshold to accept the model, so we require MSE < 0.6
            min_absolute_change=0.1,  # Minimum absolute improvement compared to baseline
            min_relative_change=0.05,  # Minimum relative improvement compared to baseline
            greater_is_better=False  # Lower MSE is better
        )
    }
    
    baseline_model_artifact_uri = mlflow.get_artifact_uri("baseline_mlflow_pyfunc")
    
    mlflow.evaluate(
        model_artifact_uri,
        test,
        targets="quality",
        model_type="regressor",
        evaluators=["default"],
        custom_metrics=[
            squared_diff_plus_one_metric,
            sum_on_target_divided_by_two_metric
        ],
        custom_artifacts=[prediction_target_scatter],
        validation_thresholds=thresholds,
        baseline_model=baseline_model_artifact_uri
    )

    print("The artifact path is",model_artifact_uri)
    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
