FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install -e .
RUN pip install -r requirements.txt

EXPOSE 8080

CMD python -m streamlit run ./src/student/app.py --server.port=8080 --server.address=0.0.0.0 & python server.py & wait
