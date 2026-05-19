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

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
classifier = pipeline("ner", model=model, tokenizer=tokenizer)

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

def convert_to_ner_tags(text, spans):
    l = ['O'] * len(text)
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    prev_end = -10
    
    for span in sorted_spans:
        end = span['end']
        start = span['start']
        label = span['label']

        for i in range(start, end):
            l[i] = "I-" + label

        if start == (prev_end + 1):
            l[start] = "B-" + label

        prev_end = end

    tokens = tokenizer(text, return_offsets_mapping=True)
    offsets = tokens["offset_mapping"]
    token_labels = []

    for (start, end) in offsets:
        if start == 0 and end == 0:
            # Special tokens like [CLS], [SEP]
            token_labels.append("O")
        else:
            token_labels.append(l[start])

    return tokens, token_labels
              
        


for poem in data:
    if poem['language'] == 'Bulgarian':
        responses = ast.literal_eval(poem["responses"])
        entities = responses["ner_tags"][0]["value"]
        for item in entities:
            item['start'] -= 1
            item['end'] -= 1

    gold_entities = responses["ner_tags"][0]["value"]



    
    
    ID2LABEL = {
    0: 'O',
    1: 'B-PER', 2: 'I-PER',
    3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC',
    7: 'B-MISC', 8: 'I-MISC',}
    LABEL2ID = {v: k for k, v in ID2LABEL.items()}
    LABELS = list(LABEL2ID.keys())

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )