import json
import ast
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

with open("testing_json.json", encoding="utf-8") as f:
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
    responses = ast.literal_eval(poem["responses"])
    entities = responses["ner_tags"][0]["value"]
    tokens, token_labels = convert_to_ner_tags(poem['text'], entities)
    print(poem['text'])
    print(tokens, token_labels)