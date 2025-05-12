FROM python:3.10


WORKDIR /app


COPY . .


RUN pip install -r requirements.txt
RUN git clone https://github.com/coudertlab/CoRE-MOF.git

EXPOSE 8080

CMD python -m streamlit run ./student/app.py --server.port=8080 --server.address=0.0.0.0 & python server.py & wait
