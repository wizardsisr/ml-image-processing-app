from fastapi import FastAPI, UploadFile, Request
from fastapi.openapi.utils import get_openapi
from app.analytics import cifar_cnn, config
import logging
from PIL import Image
import io
import json

api_app = FastAPI()


@api_app.post('/inference')
async def predict(request: Request, application: str, file: UploadFile) -> str:
    logging.info("In inference...")
    model_name, model_stage = config.model_name, config.model_stage
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    return cifar_cnn.predict(img, model_name, model_stage)


def generate_schema():
    with open('app/analytics/static/openapi.json', 'w') as f:
        json.dump(get_openapi(
            title=api_app.title,
            version=api_app.version,
            openapi_version=api_app.openapi_version,
            description=api_app.description,
            routes=api_app.routes,
        ), f, indent=4)
