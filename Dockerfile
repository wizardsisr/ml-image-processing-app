FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

WORKDIR .

COPY requirements.txt ./requirements.txt

RUN apt-get clean && apt-get update \
    && apt-get install g++ -y \
    && apt-get install gcc -y \
    && apt-get install git -y \
    && apt-get install -y default-libmysqlclient-dev \
    && apt-get clean && \
    pip3 install --no-cache-dir  -r requirements.txt

COPY app ./app

ENTRYPOINT ["streamlit", "run"]
CMD ["./app/analytics/home.py", "model_stage=Staging", "--logger.level=info", "--server.port=8080"]