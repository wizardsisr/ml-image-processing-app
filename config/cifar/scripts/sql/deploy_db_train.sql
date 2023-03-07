CREATE OR REPLACE FUNCTION run_inference_task (mlflow_stage text,
                                        git_repo text,
                                        entry_point text,
                                        experiment_name text,
                                        environment_name text,
                                        mlflow_host text,
                                        app_location text)
RETURNS TEXT
AS $$
    # container: plc_python3_shared
    import os
    import sys
    import subprocess
    import logging
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().addHandler(logging.FileHandler(f"{app_location}/debug.log"))
    import importlib
    import pkgutil
    try:
        os.environ['MLFLOW_TRACKING_URI']=mlflow_host
        os.environ['git_repo']=git_repo
        os.environ['mlflow_entry']=entry_point
        os.environ['mlflow_stage']=mlflow_stage
        os.environ['environment_name']=environment_name
        os.environ['experiment_name']=experiment_name
        os.environ['shared_app_path']=app_location
        sys.path.append(f'{app_location}/_vendor')
        sys.path.append(f'{app_location}')
        if sys.modules.get('base_app.main'):
            del sys.modules['base_app.main']
        if sys.modules.get('base_app'):
            del sys.modules['base_app']
        from base_app import main
        return subprocess.check_output('ls -ltr /', shell=True).decode(sys.stdout.encoding).strip()
    except subprocess.CalledProcessError as e:
        return e.output
$$
LANGUAGE 'plpython3u';

