#THIS IS VERSION 7 (finetuning)

    #Ideally we would do kfold across 3 runs (to take an average across 3 runs with different random seeds) but it takes a long time so I started with train test

import json
import ast
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

# new imports
#pip install datasets, pip install transformers[torch] (do these first)
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification 
from sklearn.model_selection import train_test_split
from datasets import Dataset    #HuggingFace's Dataset class, which is the format the Trainer
import torch

#FUNCTIONS ---------------------------------------------------------------------

def adjust_offsets(start, end, language):
    if language == "Bulgarian":
        return start - 1, end - 1
    else:
        return start, end


def get_span(text, start, end, language=""):
    start, end = adjust_offsets(start, end, language)
    return text[start:end]


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ner_pipeline(model_name="FacebookAI/xlm-roberta-large-finetuned-conll03-english"):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple")  #maybe we could try different versions at some point


def parse_annotations(responses_str):
    try:
        parsed = ast.literal_eval(responses_str)
        return parsed["ner_tags"][0]["value"]
    except:
        return []


def extract_text(item):
    """Handles both flat and nested text fields, and normalises line endings."""
    raw = item["text"]
    if isinstance(raw, dict):
        raw = raw["text"]               # handles {"text": {"text": "..."}}
    return raw.replace("\r\n", "\n")    # normalise Windows line endings


#Not needed now:

# def verify_offsets(data, num_examples=3, language=None):
#     print("\n--- OFFSET VERIFICATION ---")
#     checked = 0
#     for item in data:
#         if language and item["language"] != language:
#             continue
#         annotations = parse_annotations(item["responses"])
#         if not annotations:
#             continue


#         text = extract_text(item)
#         print(f"\nID: {item['id']}")
#         print(f"Language: {item['language']}")
#         for ann in annotations:
#             extracted = get_span(text, ann["start"], ann["end"], item["language"])
#             print(f"  Label: {ann['label']} | [{ann['start']}:{ann['end']}] | '{extracted}'")
#         checked += 1
#         if checked >= num_examples:
#             break
#     print("\n--- END VERIFICATION ---\n")


def spans_to_set(annotations, text, language):
    result = set()
    for a in annotations:
        start, end = adjust_offsets(a["start"], a["end"], language)
        result.add((start, end, a["label"]))
    return result


def predictions_to_set(predictions):
    return {(p["start"], p["end"], p["entity_group"]) for p in predictions}  # ← fixed typo 'pr'


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_evaluation(data, ner_pipeline):
    label_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    lang_counts  = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    overall      = {"tp": 0, "fp": 0, "fn": 0}
    skipped      = 0
    errors       = []

    total = len(data)
    for i, item in enumerate(data):
        print(f"Processing {i+1}/{total}...", end="\r")

        annotations = parse_annotations(item["responses"])
        if not annotations:
            skipped += 1
            continue

        text     = extract_text(item)   # use extract_text here
        language = item["language"]
        gold_set = spans_to_set(annotations, text, language)  # added language arg

        try:
            predictions = ner_pipeline(text)
        except Exception as e:
            errors.append({"id": item["id"], "error": str(e)})
            continue

        pred_set = predictions_to_set(predictions)
        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        overall["tp"] += len(tp)
        overall["fp"] += len(fp)
        overall["fn"] += len(fn)

        lang_counts[language]["tp"] += len(tp)
        lang_counts[language]["fp"] += len(fp)
        lang_counts[language]["fn"] += len(fn)

        for (_, _, label) in tp:
            label_counts[label]["tp"] += 1
        for (_, _, label) in fp:
            label_counts[label]["fp"] += 1
        for (_, _, label) in fn:
            label_counts[label]["fn"] += 1

    print(f"\nDone. Skipped {skipped} empty-annotation examples.")
    if errors:
        print(f"Failed on {len(errors)} examples:")
        for e in errors:
            print(f"  ID {e['id']}: {e['error']}")

    return overall, label_counts, lang_counts


def print_results(overall, label_counts, lang_counts):
    print("\n" + "="*50)
    print("OVERALL")
    print("="*50)
    p, r, f1 = compute_metrics(overall["tp"], overall["fp"], overall["fn"])
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}")

    print("\n" + "="*50)
    print("PER LABEL")
    print("="*50)
    for label, c in sorted(label_counts.items()):
        p, r, f1 = compute_metrics(c["tp"], c["fp"], c["fn"])
        print(f"  {label:6s} → P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}  "
              f"(TP={c['tp']} FP={c['fp']} FN={c['fn']})")

    print("\n" + "="*50)
    print("PER LANGUAGE")
    print("="*50)
    for lang, c in sorted(lang_counts.items()):
        p, r, f1 = compute_metrics(c["tp"], c["fp"], c["fn"])
        print(f"  {lang:10s} → P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}  "
              f"(TP={c['tp']} FP={c['fp']} FN={c['fn']})")
        
#extra functions for this: 
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}

def split_data(data, test_size=0.2, random_state=77):
    languages = [item["language"] for item in data]
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=languages,
        random_state=random_state)

    #just printing what how much is where
    print(f"Train: {len(train_data)} poems | Test: {len(test_data)} poems")
    lang_counts = defaultdict(lambda: {"train": 0, "test": 0})
    for item in train_data:
        lang_counts[item["language"]]["train"] += 1
    for item in test_data:
        lang_counts[item["language"]]["test"] += 1
    print("\nLanguage distribution:")
    for lang, counts in sorted(lang_counts.items()):
        print(f"  {lang:10s} → train: {counts['train']}  test: {counts['test']}")

    return train_data, test_data

def tokenize_and_align_labels(item, tokenizer):
    text        = extract_text(item)                    
    language    = item["language"]
    annotations = parse_annotations(item["responses"])

    # build character-level label array     (so it is the same as Roberta uses)
    char_labels = ["O"] * len(text)
    for ann in annotations:
        start, end = adjust_offsets(ann["start"], ann["end"], language)  # reuses yours
        label = ann["label"]
        for i in range(start, min(end, len(text))):
            char_labels[i] = f"B-{label}" if i == start else f"I-{label}"

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512)

    token_labels = []
    for (token_start, token_end) in encoding["offset_mapping"]:
        if token_start == token_end:  # special token
            token_labels.append(-100)
        else:
            token_labels.append(LABEL2ID.get(char_labels[token_start], LABEL2ID["O"]))

    encoding["labels"] = token_labels
    encoding.pop("offset_mapping")
    return encoding


def build_hf_dataset(data, tokenizer):
    records = []
    for item in data:
        try:
            enc = tokenize_and_align_labels(item, tokenizer)
            records.append(enc)
        except Exception as e:
            print(f"  Skipping item {item['id']}: {e}")
    return Dataset.from_list(records)


def finetune_model(train_data, model_name, output_dir="./finetuned_ner",
                   num_epochs=3, batch_size=4):
    print(f"\nLoading tokenizer and model for fine-tuning: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    print("Tokenizing training data...")
    train_dataset = build_hf_dataset(train_data, tokenizer)

    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    save_strategy="epoch",
    logging_steps=10,
    use_cpu=not torch.cuda.is_available(),  # changed from no_cuda
    report_to="none",
    seed=77)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer)

    print("\nStarting fine-tuning...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    return output_dir


#ACTUALLY RUNNING THE CODE: --------------------------------------------------------------------------------------

# if __name__ == "__main__":
JSON_PATH  = "combined_annotations_final.json"      #changed in version 4
MODEL_NAME = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
OUTPUT_DIR = "./finetuned_ner"

data = load_data(JSON_PATH)

# Split data
train_data, test_data = split_data(data, test_size=0.4, random_state=77)  # NEW
# Fine-tune on train split
finetune_model(train_data, MODEL_NAME, OUTPUT_DIR, num_epochs=3, batch_size=4)  # NEW
# Load fine-tuned model
finetuned_pipeline = load_ner_pipeline(OUTPUT_DIR)  # NEW — reuses your existing function
# Evaluate on test split — identical to what you've been running
overall, label_counts, lang_counts = run_evaluation(test_data, finetuned_pipeline)
print_results(overall, label_counts, lang_counts)




#RESULTS --------------------------------------------------------------------------------

#Language distribution (took less than a minute):
#   Bulgarian  → train: 81  test: 20
#   Danish     → train: 79  test: 20
#   Polish     → train: 76  test: 19
#   Slovak     → train: 75  test: 19


#started finetuning at 12:26 -33 gave me 2%