FROM continuumio/miniconda3

WORKDIR /app

COPY . .
RUN conda install python=3.11 -y
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda install -c conda-forge raspa2 -y
RUN pip install -e .
RUN pip install -r requirements.txt
ENV RASPA_DIR=/opt/conda

EXPOSE 8080
EXPOSE 8000

CMD python -m streamlit run ./src/student/app.py --server.port=8080 --server.address=0.0.0.0 & python server.py & uvicorn src.student.api:app --host 0.0.0.0 --port 8000 & wait