FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["bash", "-c", "python src/train.py && python src/predict.py"]
