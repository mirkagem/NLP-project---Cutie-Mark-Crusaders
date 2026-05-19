import json
import ast
import random
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

#new imports
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments,Trainer,DataCollatorForTokenClassification
import evaluate
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig

random.seed(1)
random_state = 1

# json_path  = "testing_json.json"
json_path  = "combined_annotations_trial4.json"
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
output_dir = r"C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project"

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

print(len(data))


languages = [poem["language"] for poem in data]

# splitting data
train_data, test_data = train_test_split(
    data,
    train_size=0.1,     #CHANGE THIS AFTER!
    stratify=languages,
    random_state=random_state)

train_languages = [poem["language"] for poem in train_data]

train_data, val_data = train_test_split(
    train_data,
    train_size=0.6,
    stratify=train_languages,
    random_state=random_state)

print(f"Train size : {len(train_data)}")
print(f"Validation size : {len(val_data)}")
print(f"Test size : {len(test_data)}")

# necesary stuff for model
ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABELS = list(LABEL2ID.keys())

# initializing model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True, #this should fix the mismatched labels ("B-PER" issue)
)

def build_hf_dataset(data_split, tokenizer):
    records = []
    skipped = 0
    for poem in data_split:
        try:
            text = poem["text"].replace("\r\n", "\n")
            annotations = ast.literal_eval(poem["responses"])["ner_tags"][0]["value"]
            char_labels = {}
            for ann in annotations:
                if poem['language'] == 'Bulgarian':
                    start = int(ann["start"])-1
                    end = int(ann["end"])-1
                else:          
                    start = ann["start"]
                    end   = ann["end"]
                label = ann["label"]    # e.g. "PER", "LOC"
                for i in range(start, end):
                    char_labels[i] = (label, i == start)
                
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=512, # runtime Considder chaning max_lenght
            )

            # tokens     = encoding["input_ids"]
            offsets    = encoding["offset_mapping"]
            bio_labels = []

            for (char_start, char_end) in offsets:
                if char_start == char_end:          # special token ([CLS], [SEP], etc.)
                    bio_labels.append(LABEL2ID["O"])
                    continue

                # Use the first character of the token to decide its label - need to recheck this because I believe Roberta does not start with B
                if char_start in char_labels:
                    label, is_start = char_labels[char_start]
                    prefix = "B" if is_start else "I"
                    bio_labels.append(LABEL2ID[f"{prefix}-{label}"])
                else:
                    bio_labels.append(LABEL2ID["O"])

            records.append({
                "input_ids":      encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels":         bio_labels,})
        except Exception as e:
            skipped += 1
            print(f"  Skipped poem {poem['id']}: {e}")
    print(f"  Converted {len(records)} poems ({skipped} skipped).")
    return Dataset.from_list(records)

print("Converting train split...")
train_ds = build_hf_dataset(train_data, tokenizer)
print("Converting validation split...")
val_ds = build_hf_dataset(val_data, tokenizer)
print("Converting test split...")
test_ds  = build_hf_dataset(test_data, tokenizer)

dataset = DatasetDict({"train": train_ds,"validation": val_ds, "test": test_ds})

data_collator = DataCollatorForTokenClassification(
    tokenizer,
    label_pad_token_id=-100
)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  #we could try 8 on HPC
    per_device_eval_batch_size=2,   #we could try 8 on HPC
    num_train_epochs=3,             #we could also do more epochs
    weight_decay=0.01,
    load_best_model_at_end=True,    
    metric_for_best_model="f1",
    # dataloader_num_workers=4,        # parallel data loading
    # ddp_find_unused_parameters=False # cleaner multi-GPU runs
    logging_steps=20,
    # fp16=False,              # CPU  ------ we can change parameters in this part once we do HPC (like more epochs but also specify what is our configuraion somehow)
)

def compute_metrics_hf(eval_pred):
    seqeval = evaluate.load("seqeval")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    true_labels = [
        [ID2LABEL[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    true_preds = [
        [ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_hf,
)

trainer.train()
trainer.save_model(output_dir) #saving the model
print(f"\nFine-tuned model saved to: {output_dir}")

# end of finetuning



#evaluate_finetuned(test_data)
# def evaluate_finetuned(test_data, model_dir="./finetuned-ner"):
#     print(f"\nLoading fine-tuned model from {model_dir}...")
#     ner_pipeline = load_ner_pipeline(model_name=model_dir)

#     overall, label_counts, lang_stats = run_evaluation(test_data, ner_pipeline) --- we are here
#     print_results(overall, label_counts, lang_stats)



print(f"\nLoading fine-tuned model from {model_dir}...")
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple")

label_counts = {}
lang_stats = {lang: {"tp": 0, "fp": 0, "fn": 0} for lang in set(languages)}  #make dict for that language: eg. "Bulgarian": {"tp": 0, "fp": 0, "fn": 0}
overall = {"tp": 0, "fp": 0, "fn": 0}
empty = 0                                #annotations where we didnt mark any entity
errors = []
total = len(test_data)

print('Processing of poems started.')
for i, poem in enumerate(test_data): #basically just take a poem (i is just for cosmetics)
    print(f"Processing {i+1}/{total}...", end="\r")      
    annotations = ast.literal_eval(poem["responses"])["ner_tags"][0]["value"]     #responses is one field from poem: eg. "{'ner_tags': [{'value': [{'label': 'PER', 'start': 1395, 'end': 1403}]}]}"
    if not annotations:
        empty += 1
        #countinue (if we decide not to include them)
    try:
        text = poem["text"].replace("\r\n", "\n") #we found these were creating issues
        language = poem["language"]

        predictions = ner_pipeline(text)

        gold_set = set() # spans_to_set(annotations, text, language)    #HERE we also do Bulgarian step
        for a in annotations:
            if language == "Bulgarian":
                start = a["start"] - 1 
                end = a["end"] - 1
            else:
                start = a["start"]
                end = a["end"]
            gold_set.add((start, end, a["label"]))
        pred_set = set()
        for p in predictions:
            pred_set.add((p["start"], p["end"], p["entity_group"]))
            
    except Exception as e:
        errors.append({"id": poem["id"], "error": str(e)})
        continue
    
    #calc overlap between the 2 sets
    tp = gold_set & pred_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set

    overall["tp"] += len(tp)
    overall["fp"] += len(fp)
    overall["fn"] += len(fn)
    # per language stats
    lang_stats[language]["tp"] += len(tp)
    lang_stats[language]["fp"] += len(fp)
    lang_stats[language]["fn"] += len(fn)

    #per label (PER, LOC, ORG, MISC)
    for metric_name, metric_set in [("tp", tp), ("fp", fp),("fn", fn)]:
        for (_, _, label) in metric_set:
            label_counts.setdefault(label,{"tp": 0, "fp": 0, "fn": 0})
            label_counts[label][metric_name] += 1

print(f"\nDone. There were {empty} empty-annotation examples included.")
#print any poems which got errors
for e in errors:
    print(f"  ID {e['id']}: {e['error']}")


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

print("\n")
print("OVERALL")
print("="*50)
p, r, f1 = compute_metrics(overall["tp"], overall["fp"], overall["fn"])
print(f"  Precision : {p:.3f}")     #rounded to 3 decimal places
print(f"  Recall    : {r:.3f}")
print(f"  F1        : {f1:.3f}")
print(f"  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}")

print("\n")
print("PER LABEL")
print("="*50)
for label, c in sorted(label_counts.items()):
    p, r, f1 = compute_metrics(c["tp"], c["fp"], c["fn"])
    print(f"  {label:6s} - P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")  #:6s here just adds padding for prettier printing

print("\n")
print("PER LANGUAGE")
print("="*50)
for lang, c in sorted(lang_stats.items()):
    p, r, f1 = compute_metrics(c["tp"], c["fp"], c["fn"])
    print(f"  {lang:10s} - P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")  #nicely put outcome (:10s is just padding)

#print any poems which got errors
for e in errors:
    print(f"  ID {e['id']}: {e['error']}")
