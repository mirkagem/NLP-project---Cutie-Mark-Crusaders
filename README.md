# NLP-project---Cutie-Mark-Crusaders
Named Entity Recognition in poems

## Setup
Install all needed package dependencies and download the base spaCy model using the commands:

```bash
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
```
## Run experiments

### Annotation and creation of dataset
Install Docker
Then launch an Argilla instance using Docker and ingest a target poem file:
```bash
docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart:latest
python start_argilla.py
```
After interactive text labeling at http://localhost:6900, export the finalized JSON data:
```bash
python get_annotations.py
```
When all language files are ready, combine them:
```bash
python combining_data.py
```

### RoBERTa
To evaluate zero-shot performance directly on the target dataset without training:
```bash
python Roberta_out_of_box_performance.py
```
To finetune on some part of the data and test on the rest:
```bash
python finetuning_Roberta5.py
```

### SpaCy
To both evaluate zero-shot performance directly on the target dataset without training and finetuning on some part of the data and test on the rest:
```bash
python spacy_test_and_training.py
```
