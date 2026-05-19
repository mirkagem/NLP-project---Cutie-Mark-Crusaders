import json
import ast

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
        token_labels[i] = l[start]
        if start == 0 and end == 0:
            # Special tokens like [CLS], [SEP]
            token_labels.append("O")
        else:
            token_labels.append(l[start])

    return tokens, token_labels