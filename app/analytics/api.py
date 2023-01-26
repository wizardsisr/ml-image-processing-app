from fastapi import FastAPI, UploadFile, Request
from app.analytics import cifar_cnn, config
import logging
from PIL import Image
import io

app = FastAPI()


@app.post('/inference')
async def predict(request: Request, application: str, file: UploadFile) -> str:
    logging.info("In inference...")
    model_name, model_stage = config.model_name, config.model_stage
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    return cifar_cnn.predict(img, model_name, model_stage)
