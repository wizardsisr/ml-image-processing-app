import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().addHandler(logging.FileHandler(f"app.log"))

from app.analytics import preloader, cifar_cnn
from tensorflow.keras.preprocessing.image import img_to_array
import greenplumpython
from pyservicebinding import binding
sb = binding.ServiceBinding()


# ## Upload dataset

# Upload dataset to S3 via MlFlow
def upload_dataset(dataset, dataset_url=None):
    return cifar_cnn.upload_dataset(dataset, dataset_url)


# ## Download DataSet
def download_dataset(artifact):
    return cifar_cnn.download_dataset(artifact)


# ## Train Model
def train_model(model_name, model_flavor, model_stage, data, epochs=10):
    return cifar_cnn.train_model(model_name, model_flavor, model_stage, data, epochs)


# ## Evaluate Model
def evaluate_model(model_name, model_flavor):
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
def predict(img, model_name, model_stage, schema='public'):
    # Resize / normalize the image
    img = img.resize((32, 32))
    img = img_to_array(img)
    img = img.reshape(-1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0

    # Get a handle for the Greenplum inference function
    inference_function = greenplumpython.function('run_inference_task', schema=schema)
    bindings = _get_bindings("greenplum", "vmware")
    url = f"postgresql://{bindings.get('username')}:{bindings.get('password')}@{bindings.get('host')}:{bindings.get('port')}/{bindings.get('database')}?sslmode=require"
    logging.info(f"JDBC URL: {url}")
    db = greenplumpython.database(uri=url)

    # Invoke Greenplum function
    # TODO: Retrieve arguments from config
    result = db.apply(lambda: inference_function(img, model_name, model_stage, 'mlapp', 'http://mlflow.tanzumlai.com',
                                                 'https://github.com/agapebondservant/ml-image-processing-app.git',
                                                 'gp-main'))
    logging.info(f"Result = {result}")
    return result


def _get_bindings(service_type, service_provider):
    global sb
    bindings_list = sb.bindings(service_type, service_provider)
    return bindings_list[0] if len(bindings_list) else {}

