import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().addHandler(logging.FileHandler(f"app.log"))

from app.analytics import preloader, cifar_cnn, config
import greenplumpython
from pyservicebinding import binding
import pickle
import os

# Service Bindings
sb = binding.ServiceBinding()
bindings = next(iter(sb.bindings("postgres", "vmware") or []), {})

# GreenplumPython
db = greenplumpython.database(uri=f"postgresql://{bindings.get('username')}:"
                                  f"{bindings.get('password')}@{bindings.get('host')}:"
                                  f"{bindings.get('port')}/{bindings.get('database')}?sslmode=require")
inference_function_name = 'run_inference_task'
inference_function = greenplumpython.function(inference_function_name, schema=os.environ['DATA_E2E_MLAPP_INFERENCE_DB_SCHEMA'])


# ## Upload dataset

# Upload dataset to S3 via MlFlow
def upload_dataset(dataset, dataset_url=None):
    """
    Uploads the dataset.
    """
    return cifar_cnn.upload_dataset(dataset, dataset_url, to_parquet=True)


# ## Download DataSet
def download_dataset(artifact):
    """
    Downloads the dataset.
    """
    return cifar_cnn.download_dataset(artifact)


# ## Train Model
def train_model(model_name, model_flavor, model_stage, data, epochs=10):
    """
    Performs training on the provided CNN model.
    """
    return cifar_cnn.train_model(model_name, model_flavor, model_stage, data, epochs)


# ## Evaluate Model
def evaluate_model(model_name, model_flavor):
    """
    Evaluates the performance of the model based on specified criteria.
    """
    return cifar_cnn.evaluate_model(model_name, model_flavor)


# ## Promote Model to Staging
def promote_model_to_staging(base_model_name, candidate_model_name, evaluation_dataset_name, model_flavor,
                             use_prior_version_as_base=False):
    """
    Evaluates the performance of the currently trained candidate model compared to the base model.
    The model that performs better based on specific metrics is then promoted to Staging.
    """
    return cifar_cnn.promote_model_to_staging(base_model_name, candidate_model_name, evaluation_dataset_name,
                                              model_flavor,
                                              use_prior_version_as_base)


# ## Make Prediction
def predict(img, model_name, model_stage):
    """
    Returns a CIFAR 10 class label representing the model's classification of an image.
    """
    img = pickle.dumps(img)

    # Invoke Greenplum function
    df = db.apply(lambda: inference_function(img, model_name, model_stage,
                                             config.plpython_base_dir,
                                             os.environ['MLFLOW_TRACKING_URI'],
                                             os.environ['DATA_E2E_MLAPP_GIT_REPO'],
                                             os.environ['DATA_E2E_MLAPP_GIT_REPO_BRANCH']))
    result = next(iter(df))[inference_function_name]
    logging.info(f"Result = {result}")
    return result
