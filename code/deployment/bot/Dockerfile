FROM python:3.11

ENV PYTHONUNBUFFERED=1

COPY /code/deployment/bot/requirements.txt /requirements.txt
COPY /code/deployment/bot/bot.py /bot.py
RUN pip install --upgrade -r /requirements.txt

EXPOSE 80
# Command to run the Streamlit app
CMD ["python", "/bot.py"]