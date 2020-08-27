# Voice2sql

A project of getting SQL query from a speech in natural language. A goal will be achieved by implementing
* Noise-robust Automatic Speech Recognition engine.
* Natual language to SQL engine.
* Table-aware name correction.

## Model
-----------------------------
### Voice to Natural Language Text
**Background**:
- https://github.com/lee-junseok/Membrane
- https://github.com/mozilla/DeepSpeech

A noise-robust Automatic Speech Recognition engine implementing the denoising function from [Membrane Project](https://github.com/lee-junseok/Membrane) and [DeepSpeech](https://github.com/mozilla/DeepSpeech) speech recognition model.

An example of asking *"what was the result of the game with New York Jets?"*:

![ASR Screen Shot](static/ASR_screen_shot.png)

- Proper name would not be transcribed exactly.
- Table-aware name correction will handle this.
-------------------------------------
### Natural Language to SQL
**Background**:
- https://github.com/donglixp/coarse2fine
- https://github.com/prezaei85/nl2sql
- https://github.com/salesforce/WikiSQL

***must need GPU***

A natiural language to SQL model trained on the [WikiSQL](https://github.com/salesforce/WikiSQL) table data.
------------------------------
## Installation
* To install the model:
```
bash setup.sh
```
*for Mac users*:
```
bash setup_mac.sh
```
If you want to install it manually here is what `setup.sh` looks like:
```
# Setup virtual environment
conda create -n voice2sql python=3.6 pip
conda activate voice2sql

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

```

## Todo


- Noise-robust ASR
- Table-aware word correction
- NL2SQL

***Updating***
