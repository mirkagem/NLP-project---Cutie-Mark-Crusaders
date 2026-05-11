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
    p = m["correct"] / m["predicted"]
    r = m["correct"] / m["gold"]
    f1 = 2 * p * r / (p + r)
    return p, r, f1

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

metrics = defaultdict(lambda: {
    "correct": 0,
    "predicted": 0,
    "gold": 0
})

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

    m = metrics[language]
    m["correct"] += matched
    m["predicted"] += len(pred_entities)
    m["gold"] += len(gold_entities)

    m_all = metrics["all"]
    m_all["correct"] += matched
    m_all["predicted"] += len(pred_entities)
    m_all["gold"] += len(gold_entities)

print("\nPerformance on all data before any additional training: \n")

print(f"{'Language':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 50)

p, r, f1 = compute(metrics["all"])
print(f"{'Overall':<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")

for language, m in metrics.items():

    if language == "all":
        continue

    p, r, f1 = compute(m)

    print(f"{language:<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")


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

trained_metrics = defaultdict(lambda: {
    "correct": 0,
    "predicted": 0,
    "gold": 0
})

for text, annotations, language in test_data:

    doc = nlp(text)

    pred_entities = [
        (ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
    ]

    gold_entities = annotations["entities"]
    
    matched = 0
    
    for g in gold_entities:
        for p in pred_entities:
            if g == p:
                matched += 1
                break

    trained_metrics[language]["correct"] += matched
    trained_metrics[language]["predicted"] += len(pred_entities)
    trained_metrics[language]["gold"] += len(gold_entities)

    trained_metrics["all"]["correct"] += matched
    trained_metrics["all"]["predicted"] += len(pred_entities)
    trained_metrics["all"]["gold"] += len(gold_entities)

print("\nPerformance on 40 percent of data after additional training:\n")

print(f"{'Language':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 50)

p, r, f1 = compute(trained_metrics["all"])
print(f"{'Overall':<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")

for language, m in trained_metrics.items():

    if language == "all":
        continue

    p, r, f1 = compute(m)

    print(f"{language:<15} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")