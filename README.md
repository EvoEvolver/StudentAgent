# StudentAgent

The agent who can learn.

## Installation

Install RASPA2 as explained on their repo.
For example, use:
```bash
conda install -c conda-forge raspa2
export RASPA_DIR=/path/to/environment/root
```


```bash
pip install -r requirements.txt
pip install -e .
```
Git clone CoRE-MOF/ repository (https://github.com/coudertlab/CoRE-MOF) and install with pip.

## Setup
```bash
Create a .env file containing:
OPENAI_API_KEY="..."
RASPA_DIR="..."
TEMP_PATH = "test/"
```

## Usage

```bash
python -m streamlit run student/app.py
```
