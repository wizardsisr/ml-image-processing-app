echo "SELECT run_training_task(:'mlflow_stage', :'git_repo', :'entry_point', :'experiment_name', :'environment_name', :'mlflow_host', :'app_location')" \
  | psql -d dev -U gpadmin -h ${GREENPLUM_MASTER} \
         -v mlflow_stage="${MLFLOW_STAGE}" \
         -v git_repo="${GIT_REPO}" \
         -v entry_point="${ENTRY_POINT}" \
         -v experiment_name="${EXPERIMENT_NAME}" \
         -v environment_name="${ENVIRONMENT_NAME}" \
         -v mlflow_host="${MLFLOW_TRACKING_URI}" \
         -v app_location="${PYFUNC_APP_LOCATION}";