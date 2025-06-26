FROM continuumio/miniconda3

WORKDIR /app

COPY . .

RUN conda install python=3.11 -y
RUN conda install -c conda-forge raspa2 -y
RUN pip install -e .
RUN pip install -r requirements.txt
ENV RASPA_DIR=/opt/conda

EXPOSE 8080

CMD python -m streamlit run ./src/student/app.py --server.port=8080 --server.address=0.0.0.0 & python server.py & wait