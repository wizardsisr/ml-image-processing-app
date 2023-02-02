import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
import tensorflow
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import logging
import pathlib
import mlflow
from mlflow import MlflowClient
import warnings
import tarfile
import sys
import importlib
import hickle as hkl
import os
import time
import logging
import traceback
from io import StringIO
import http
import requests
from PIL import Image

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')


# ## Utilities
def get_run_for_artifacts(active_run_id):
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default'
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.mainartifacts='y'", max_results=1,
                              output_format='list')
    if len(runs):
        return runs[0].info.run_id
    else:
        mlflow.set_tags({'mainartifacts': 'y'})
        return active_run_id


def get_root_run(active_run_id):
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default'
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.runlevel='root'", max_results=1,
                              output_format='list')
    if len(runs):
        parent_run_id = runs[0].info.run_id
        mlflow.set_tags({'mlflow.parentRunId': parent_run_id})
        return parent_run_id
    else:
        mlflow.set_tags({'runlevel': 'root'})
        return active_run_id


def get_current_run():
    last_active_run = mlflow.last_active_run()
    return last_active_run.info.run_id if last_active_run else None


# ## Upload dataset

# Upload dataset to S3 via MlFlow
def upload_dataset(dataset, dataset_url=None):
    experiment_id = mlflow.get_experiment_by_name(os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default')
    with mlflow.start_run(run_name='upload_dataset') as active_run:
        artifact_run_id = get_run_for_artifacts(active_run.info.run_id)

        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT = '3600'
        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '3600'

        logging.info(f'Artifact run id is {artifact_run_id}')

        client = MlflowClient()

        mod = importlib.import_module(f'tensorflow.keras.datasets.{dataset}')
        (training_data, training_labels), (test_data, test_labels) = mod.load_data()
        training_data, test_data = training_data / 255.0, test_data / 255.0

        hkl.dump({'training_data': training_data,
                  'test_data': test_data,
                  'training_labels': training_labels,
                  'test_labels': test_labels},
                 dataset,
                 compression='gzip',
                 mode='w')

        try:
            client.log_artifact(artifact_run_id, dataset)
            logging.info(f'File uploaded - run id {artifact_run_id}')
        except Exception as e:
            logging.error(f'Could not complete upload for run id {artifact_run_id} - error occurred: ', exc_info=True)
            traceback.print_exc()


# ## Download DataSet
def download_dataset(artifact):
    try:
        with mlflow.start_run(run_name='download_dataset', nested=True) as active_run:
            artifact_run_id = get_run_for_artifacts(active_run.info.run_id)
            with mlflow.start_run(run_id=artifact_run_id, nested='True'):
                uri = mlflow.get_artifact_uri(artifact_path=artifact)
                uri = uri.replace('mlflow-artifacts:', f'{os.environ.get("MLFLOW_S3_ENDPOINT_URL")}/mlflow/artifacts')
                # uri = uri.replace('mlflow-artifacts:',f'http:/{mlflow.get_tracking_uri()}/api/2.0/mlflow-artifacts/artifacts')
                logging.info(f'Download uri: {uri}')
                req = requests.get(uri)
                with open(f'downloads/{artifact}', 'wb') as f:
                    f.write(req.content)
                    logging.info(f'{artifact} download complete.')
                return f'downloads/{artifact}'
    except http.client.IncompleteRead as icread:
        logging.info(f'Incomplete read...{icread}')
    except Exception as e:
        logging.error(f'Could not complete upload for run id - error occurred: ', exc_info=True)
        traceback.print_exc()


def download_model(model_name, model_flavor, best_run_id, retries=2):
    with mlflow.start_run(run_name='download_model', nested=True) as active_run:
        try:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}' and run_id='{best_run_id}'")
            logging.info(f"In download model...search model results = {versions}")
            if len(versions) and versions[0].current_stage.lower() != 'production':
                version = versions[0].version
                model = getattr(mlflow, model_flavor).load_model(f'models:/{model_name}/{version}')
                return model, version
            else:
                logging.info(f"No suitable candidate model found for {model_name}...")
                return None
        except Exception as e:
            if retries > 0:
                logging.error(f"Could not download {model_name} - retrying...")
                time.sleep(1)
                download_model(model_name, model_flavor, best_run_id, retries=retries - 1)
            else:
                logging.error(f'Could not complete download for model {model_name} - error occurred: ', exc_info=True)
                traceback.print_exc()


# ## Train Model
def train_model(model_name, model_flavor, model_stage, data, epochs=10):
    with mlflow.start_run(run_name='train_model', nested=True) as active_run:
        # Build and Compile Model
        with tensorflow.device('/CPU:0'):  # Place tensors on the CPU
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.summary()
            model.compile(optimizer='adam',
                          loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            # Fit model
            history = model.fit(data.get('training_data'), data.get('training_labels'), epochs=epochs,
                                validation_data=(data.get('test_data'), data.get('test_labels')))

        # Plot metrics
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.savefig("cifar_cnn_accuracy.png", bbox_inches='tight')
        mlflow.log_artifact("cifar_cnn_accuracy.png")

        # Log Metrics
        test_loss, test_acc = model.evaluate(data.get('test_data'), data.get('test_labels'), verbose=2)
        mlflow.log_metric('testing_loss', test_loss)
        mlflow.log_metric('testing_accuracy', test_acc)
        getattr(mlflow, model_flavor).autolog(log_models=False)

        getattr(mlflow, model_flavor).log_model(model,
                                                artifact_path=f'runs:/{active_run.info.run_id}/{model_name}',
                                                registered_model_name=model_name,
                                                await_registration_for=None)

        return model


# ## Evaluate Model
def evaluate_model(model_name, model_flavor):
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default'

    with mlflow.start_run(run_name='evaluate_model', nested=True) as active_run:
        runs = mlflow.search_runs(experiment_names=[experiment_name],
                                  filter_string="attributes.run_name='train_model'",
                                  order_by=['metrics.testing_accuracy DESC'],
                                  max_results=1,
                                  output_format='list')
        logging.info(f"Best run found is...{runs}")
        best_run_id = runs[0].info.run_id if len(runs) else None
        if best_run_id is not None:
            (model, version) = download_model(model_name, model_flavor, best_run_id)
            logging.info(f"Found best model: {model}")

            MlflowClient().transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )


# ## Make Prediction
def predict(img, model_name, model_stage):
    try:
        model = getattr(mlflow, 'tensorflow').load_model(f'models:/{model_name}/{model_stage}')
    except Exception as e:
        logging.info('Could not load model at this time.')
        return None

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0

    prediction_results = model.predict(img)
    prediction = labels[np.argmax(prediction_results)]
    logging.info(
        f'Predictions in order...list={prediction_results}, index={np.argmax(prediction_results)}, prediction={prediction}')
    return prediction
