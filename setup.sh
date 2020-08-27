#!/bin/bash
# Setup virtual environment
conda create -n voice2sql python=3.6 pip
conda activate voice2sql
# python3 -m venv voice2sql
# source voice2sql/bin/activate

# Install packages
python -m pip install -r requirements.txt
wget http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
python -m pip install torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
python -m spacy download en_core_web_sm
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
unzip stanford-corenlp-full-2017-06-09.zip
export CORENLP_HOME=~/nl2sql/stanford-corenlp-full-2017-06-09

# Pretrained NL2SQL model
wget https://voice2sql.s3-us-west-2.amazonaws.com/pretrained.tar.gz
tar -xzvf pretrained.tar.gz

# Pretrained Voice-to-text model
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.pbmm -P deepspeech_data/
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.scorer -P deepspeech_data/
