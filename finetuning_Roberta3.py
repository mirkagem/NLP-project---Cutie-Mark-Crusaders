import json
import ast
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

#new imports
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments,Trainer,DataCollatorForTokenClassification
import evaluate
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig

json_path  = "combined_annotations_final2.json"
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

for poem in data:
    if poem['language'] == 'Bulgarian':
        for span in poem['responses']:
            span['start'] -= 1
            span['start'] -= 1