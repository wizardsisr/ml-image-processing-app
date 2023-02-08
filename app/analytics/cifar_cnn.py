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
from sklearn.dummy import DummyClassifier
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
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
import json
from PIL import Image
from mlflow.models import MetricThreshold
from app.analytics import mlflow_utils
from evidently.test_suite import TestSuite
from evidently.test_preset import MulticlassClassificationTestPreset

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')


# ## Upload dataset

# Upload dataset to S3 via MlFlow
def upload_dataset(dataset, dataset_url=None):
    with mlflow.start_run(run_name='upload_dataset', nested=True) as active_run:
        mlflow_utils.prep_mlflow_run(active_run)
        artifact_run_id = mlflow_utils.get_root_run(active_run.info.run_id)
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
            mlflow_utils.prep_mlflow_run(active_run)
            artifact_run_id = mlflow_utils.get_root_run(active_run.info.run_id)
            with mlflow.start_run(run_id=artifact_run_id, nested='True'):
                uri = mlflow.get_artifact_uri(artifact_path=artifact)
                uri = uri.replace('mlflow-artifacts:', f'{os.environ.get("MLFLOW_S3_ENDPOINT_URL")}/mlflow/artifacts')
                logging.info(f'Download uri: {uri}')
                req = requests.get(uri)
                download_path = f'downloads/{artifact}'
                os.makedirs(os.path.dirname(f'downloads/{artifact}'), exist_ok=True)
                with open(download_path, 'wb') as f:
                    f.write(req.content)
                    logging.info(f'{artifact} download complete: {download_path}')
                return download_path
    except http.client.IncompleteRead as icread:
        logging.info(f'Incomplete read...{icread}')
    except Exception as e:
        logging.error(f'Could not complete upload for run id - error occurred: ', exc_info=True)
        traceback.print_exc()


def download_model(model_name, model_flavor, best_run_id=None, retries=2):
    model, version = None, 0
    with mlflow.start_run(run_name='download_model', nested=True) as active_run:
        mlflow_utils.prep_mlflow_run(active_run)
        try:
            client = MlflowClient()
            if best_run_id:
                versions = client.search_model_versions(f"name='{model_name}' and run_id='{best_run_id}'")
            else:
                versions = client.get_latest_versions(model_name)
            logging.info(f"In download model...search model results = {versions}")
            if len(versions) and versions[0].current_stage.lower() != 'production':
                version = versions[0].version
                model = getattr(mlflow, model_flavor).load_model(f'models:/{model_name}/{version}')
            else:
                logging.info(f"No suitable candidate model found for {model_name}...")
        except Exception as e:
            if retries > 0:
                logging.error(f"Could not download {model_name} - retrying...")
                time.sleep(10)
                download_model(model_name, model_flavor, best_run_id=best_run_id, retries=retries - 1)
            else:
                logging.error(f'Could not complete download for model {model_name} - error occurred: ', exc_info=True)
        return model, version


# ## Train Model
def train_model(model_name, model_flavor, model_stage, data, epochs=10):
    with mlflow.start_run(run_name='train_model', nested=True) as active_run:
        mlflow_utils.prep_mlflow_run(active_run)
        artifact_run_id = mlflow_utils.get_root_run(active_run.info.run_id)
        client = MlflowClient()
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
        client.log_artifact(artifact_run_id, "cifar_cnn_accuracy.png")

        # Log Metrics
        test_loss, test_acc = model.evaluate(data.get('test_data'), data.get('test_labels'), verbose=2)
        client.log_metric(artifact_run_id, 'testing_loss', test_loss)
        client.log_metric(artifact_run_id, 'testing_accuracy', test_acc)
        getattr(mlflow, model_flavor).autolog(log_models=False)

        # Register model
        getattr(mlflow, model_flavor).log_model(model,
                                                artifact_path=model_name,
                                                registered_model_name=model_name)

        return model


# ## Evaluate Model
def evaluate_model(model_name, model_flavor):
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default'

    with mlflow.start_run(run_name='evaluate_model', nested=True) as active_run:
        mlflow_utils.prep_mlflow_run(active_run)
        best_runs = mlflow.search_runs(experiment_names=[experiment_name],
                                       filter_string="attributes.run_name='train_model'",
                                       order_by=['metrics.testing_accuracy DESC'],
                                       max_results=1,
                                       output_format='list')
        best_run_id = best_runs[0].info.run_id if len(best_runs) else None

        if best_run_id is not None:
            (model, version) = download_model(model_name, model_flavor, best_run_id=best_run_id)
            logging.info(f"Found best model for experiments {experiment_name}, model name {model_name} : {model}")
            MlflowClient().transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )


# ## Promote Model to Staging
def promote_model_to_staging(base_model_name, candidate_model_name, evaluation_dataset_name, model_flavor):
    """
    Evaluates the performance of the currently trained candidate model compared to the base model.
    The model that performs better based on specific metrics is then promoted to Staging.
    """

    with mlflow.start_run(run_name='promote_model_to_staging', nested=True) as active_run:
        mlflow_utils.prep_mlflow_run(active_run)
        artifact_run_id = mlflow_utils.get_root_run(active_run.info.run_id)
        client = MlflowClient()

        _data_path = download_dataset(evaluation_dataset_name)
        _data = hkl.load(_data_path)
        (candidate_model, candidate_model_version) = download_model(candidate_model_name, model_flavor, retries=6)
        (base_model, base_model_version) = download_model(base_model_name, model_flavor, retries=6)
        test_data = _data.get('test_data')
        test_labels = _data.get('test_labels')
        preexisting_base_model_found = base_model is not None

        if not preexisting_base_model_found:
            logging.info(f"No prior base model found with name {base_model_name}; preparing dummy model...")
            size, num_classes = test_labels.shape[0], 10
            dummy_data = pd.DataFrame({'x': np.random.randint(0, num_classes, size),
                                       'y': test_labels.reshape(size, )})
            base_model = DummyClassifier(strategy="uniform").fit(dummy_data['x'], dummy_data['y'])

        # Generate and Save Evaluation Metrics
        curr_data = _tensors_to_1d_prediction_and_target(test_data, test_labels, candidate_model)
        ref_data = _tensors_to_1d_prediction_and_target(test_data, test_labels, base_model)
        tests = TestSuite(tests=[
            MulticlassClassificationTestPreset()
        ])
        tests.run(current_data=curr_data, reference_data=ref_data)

        # Log Evaluation Metrics
        tests_results_json = tests.json()
        logging.info(f"Evidently generated results...{tests_results_json}")

        # Save Evaluation Metrics Report
        tests.save_html('/tmp/test_results.html')
        client.log_artifact(artifact_run_id, "/tmp/test_results.html")
        logging.info("Model evaluation report generated successfully.")

        # Determine the best model
        tests_results_json_tests = json.loads(tests_results_json)['tests']
        accuracy_results = np.array([test for test in tests_results_json_tests if test['name'] == 'Accuracy Score'])
        if len(accuracy_results):
            client.log_metric(artifact_run_id, 'accuracy_score', accuracy_results[0]['parameters']['accuracy'])
            promote_candidate_model = accuracy_results[0]['status'] == 'SUCCESS' or not preexisting_base_model_found

            # Promote the best model
            getattr(mlflow, model_flavor).log_model(candidate_model if promote_candidate_model else base_model,
                                                    artifact_path=base_model_name,
                                                    registered_model_name=base_model_name)

            client.transition_model_version_stage(
                name=base_model_name,
                version=base_model_version + 1,
                stage="Staging"
            )
        else:
            raise Exception("ERROR: Failed to identify model for promotion.")


# ## Make Prediction
def predict(img, model_name, model_stage):
    try:
        model = getattr(mlflow, 'tensorflow').load_model(f'models:/{model_name}/{model_stage}')
    except Exception as e:
        logging.info(f'Could not load model at this time: for model name={model_name}, model stage={model_stage}')
        traceback.print_exc()
        return None

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    img = img.resize((32, 32))
    img = img_to_array(img)
    img = img.reshape(-1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0

    prediction_results = model.predict(img)
    prediction = labels[np.argmax(prediction_results)]
    logging.info(
        f'Predictions in order...list={prediction_results}, index={np.argmax(prediction_results)}, prediction={prediction}')
    return prediction


def _tensors_to_1d_prediction_and_target(tensor_data, tensor_labels, tensor_model):
    predictions = tensor_model.predict(tensor_data)
    predictions = predictions.reshape(predictions.shape[0], -1)
    data = pd.DataFrame({'prediction': np.fromiter((np.argmax(x) for x in predictions), dtype='int'),
                         'target': tensor_labels.reshape(tensor_labels.shape[0], )})
    return data
