import spacy
import json
import ast
import random
from collections import defaultdict
from spacy.training import Example
from spacy.util import fix_random_seed

random.seed(1)
fix_random_seed(1)

nlp = spacy.load("xx_ent_wiki_sm")

def compute(m):
    p = m["correct"] / m["predicted"] if m["predicted"] > 0 else 0.0
    r = m["correct"] / m["gold"] if m["gold"] > 0 else 0.0

    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def print_metrics_table(metrics_dict, title, key_header):
    print(f"\n--- {title} ---")
    print(f"{key_header:<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    
    for key, m in metrics_dict.items():
        if key == "all":
            continue
        p, r, f1 = compute(m)
        print(f"{key:<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")
        
    print("-" * 50)
    # Print overall summary row
    if "all" in metrics_dict:
        p, r, f1 = compute(metrics_dict["all"])
        print(f"{'All':<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f}\n")

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

lang_metrics_before = defaultdict(lambda: {"correct": 0, "predicted": 0, "gold": 0})
label_metrics_before = defaultdict(lambda: {"correct": 0, "predicted": 0, "gold": 0})
by_language = defaultdict(list)

for item in data:
    language = item["language"]
    text = item["text"]
    text = text.replace("\r", "")

    responses = ast.literal_eval(item["responses"])
    gold_entities = responses["ner_tags"][0]["value"]

    doc = nlp(text)
    pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


    if language == 'Bulgarian':
        gold_entities = [(e["start"] - 1, e["end"] - 1, e["label"]) for e in gold_entities] 
    else:
        gold_entities = [(e["start"], e["end"], e["label"]) for e in gold_entities]


    by_language[language].append((text, {"entities": list(gold_entities)}, language))

    matched = 0
    for g in gold_entities:
        for p in pred_entities:
            if g == p:
                matched += 1
                break

    for g in gold_entities:
        g_label = g[2]
        
        lang_metrics_before[language]["gold"] += 1
        lang_metrics_before["all"]["gold"] += 1
        
        label_metrics_before[g_label]["gold"] += 1
        label_metrics_before["all"]["gold"] += 1
        
        if g in pred_entities:
            lang_metrics_before[language]["correct"] += 1
            lang_metrics_before["all"]["correct"] += 1
            
            label_metrics_before[g_label]["correct"] += 1
            label_metrics_before["all"]["correct"] += 1

    for p in pred_entities:
        p_label = p[2]
        
        lang_metrics_before[language]["predicted"] += 1
        lang_metrics_before["all"]["predicted"] += 1
        
        label_metrics_before[p_label]["predicted"] += 1
        label_metrics_before["all"]["predicted"] += 1

print("\nPerformance on all data before any additional training: \n")

print_metrics_table(lang_metrics_before, "Performance per language", "Language")
print_metrics_table(label_metrics_before, "Performance per label", "Label")

print("\nTraining...\n")

train_data = []
test_data = []

for language, items in by_language.items():
    random.shuffle(items)
    split = (len(items) // 5) * 3
    train_data.extend(items[:split])
    test_data.extend(items[split:])

other_pipes = [p for p in nlp.pipe_names if p != "ner"]

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for epoch in range(10):
        random.shuffle(train_data)
        losses = {}
        for text, annotations, language in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
                [example],
                drop=0.3,
                losses=losses
            )
        print(f"Epoch {epoch+1} {losses}")

lang_metrics_after = defaultdict(lambda: {"correct": 0, "predicted": 0, "gold": 0})
label_metrics_after = defaultdict(lambda: {"correct": 0, "predicted": 0, "gold": 0})

for text, annotations, language in test_data:

    doc = nlp(text)

    pred_entities = [
        (ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
    ]

    gold_entities = annotations["entities"]
    
    for g in gold_entities:
        g_label = g[2]
        
        lang_metrics_after[language]["gold"] += 1
        lang_metrics_after["all"]["gold"] += 1
        
        label_metrics_after[g_label]["gold"] += 1
        label_metrics_after["all"]["gold"] += 1
        
        if g in pred_entities:
            lang_metrics_after[language]["correct"] += 1
            lang_metrics_after["all"]["correct"] += 1
            
            label_metrics_after[g_label]["correct"] += 1
            label_metrics_after["all"]["correct"] += 1

    for p in pred_entities:
        p_label = p[2]
        
        lang_metrics_after[language]["predicted"] += 1
        lang_metrics_after["all"]["predicted"] += 1
        
        label_metrics_after[p_label]["predicted"] += 1
        label_metrics_after["all"]["predicted"] += 1

print("\nPerformance on 40 percent of data after additional training:\n")

print_metrics_table(lang_metrics_after, "Performance per language", "Language")
print_metrics_table(label_metrics_after, "" \
"Performance per label", "Label")
