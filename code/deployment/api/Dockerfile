FROM python:3.11

ENV SPAM_MODEL_PATH=/SpamDetector
ENV SENTIMENT_MODEL_PATH=/LSTM
ENV TEMP_VOICE_PATH=/files
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /files
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY /code/deployment/api/requirements.txt /requirements.txt
COPY /code/deployment/api/api.py /api.py

COPY models/LSTM /LSTM
COPY models/SpamDetector /SpamDetector



RUN pip install --upgrade -r /requirements.txt

EXPOSE 5000
# Run the FastAPI server
ENV FLASK_APP=/api.py

# Use the flask command to run the server
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]