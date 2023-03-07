CREATE EXTENSION 'plpython3u';
CREATE OR REPLACE FUNCTION run_inference_task (model_name text, model_stage text)
RETURNS TEXT
AS $$
    # container: plc_python3_shared
    import os
    import sys
    import subprocess
    import logging
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().addHandler(logging.FileHandler(f"{app_location}/debug.log"))
    try:
        sys.path.append(f'{app_location}/_vendor')
        sys.path.append(f'{app_location}')
        if sys.modules.get('app.analytics.cifar_cnn'):
            del sys.modules['app.analytics.cifar_cnn']
        if sys.modules.get('app.analytics.config'):
            del sys.modules['app.analytics.config']
        from app.analytics import cifar_cnn, config
        return cifar_cnn.predict(img, model_name, model_stage)
    except subprocess.CalledProcessError as e:
        return e.output
$$
LANGUAGE 'plpython3u';
