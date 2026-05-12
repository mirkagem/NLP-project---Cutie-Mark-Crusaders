#pip install torch transformers datasets scikit-learn evaluate seqeval accelerate

import json
import ast
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

#new imports
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments,Trainer,DataCollatorForTokenClassification
import evaluate
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig

def load_ner_pipeline(model_name="FacebookAI/xlm-roberta-large-finetuned-conll03-english"):
    '''Building NER pipeline using the model - taken from Roberta's guide how to use it'''
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple")

def run_evaluation(data, ner_pipeline):
    label_counts = {}
    languages = ["Bulgarian", "Slovak", "Polish", "Danish"]
    lang_stats = {lang: {"tp": 0, "fp": 0, "fn": 0} for lang in languages}  #make dict for that language: eg. "Bulgarian": {"tp": 0, "fp": 0, "fn": 0}
    overall = {"tp": 0, "fp": 0, "fn": 0}
    empty = 0                                #annotations where we didnt mark any entity
    errors = []
    total = len(data)

    print('Processing of poems started.')
    for i, poem in enumerate(data): #basically just take a poem (i is just for cosmetics)
        print(f"Processing {i+1}/{total}...", end="\r")      
        annotations = ast.literal_eval(poem["responses"])["ner_tags"][0]["value"]     #responses is one field from poem: eg. "{'ner_tags': [{'value': [{'label': 'PER', 'start': 1395, 'end': 1403}]}]}"
        if not annotations:
            empty += 1
            #countinue (if we decide not to include them)
        try:
            text = poem["text"].replace("\r\n", "\n") #we found these were creating issues
            language = poem["language"]

            predictions = ner_pipeline(text)
            gold_set = spans_to_set(annotations, text, language)    #HERE we also do Bulgarian step
            pred_set = predictions_to_set(predictions)

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
    return overall, label_counts, lang_stats

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def print_results(overall, label_counts, lang_counts):
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
    for lang, c in sorted(lang_counts.items()):
        p, r, f1 = compute_metrics(c["tp"], c["fp"], c["fn"])
        print(f"  {lang:10s} - P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")  #nicely put outcome (:10s is just padding)

#2 functions to make sets (as sets are later used to calc TP, TN, FP) + offsetting fix
def spans_to_set(annotations, text, language):
    '''For our gold standard: {"label": "PER","start": 347,"end": 352} ---> (347, 352, "PER"]) + adjust for Bulgarian'''
    result = set()
    for a in annotations:
        start, end = adjust_offsets(a["start"], a["end"], language)     #needed to correct the Bulgarian entities
        result.add((start, end, a["label"]))
    return result
def predictions_to_set(predictions):
    '''For Robertas predictions: {"entity_group": "PER","start": 347,"end": 352} ----> (347, 352, "PER"])'''
    result = set()
    for p in predictions:
        result.add((p["start"], p["end"], p["entity_group"]))
    return result
def adjust_offsets(start, end, language):
    #separate function in case we later need to correct something else
    if language == "Bulgarian":
        return start - 1, end - 1
    else:
        return start, end

#NEW FUNCTIONS =============================================================================
def stratified_split(data, train_size=0.6, random_state=42):
    languages = [poem["language"] for poem in data]

    train_data, test_data = train_test_split(
        data,
        train_size=train_size,
        stratify=languages,
        random_state=random_state)

    return train_data, test_data

def print_language_distribution(dataset, name): #I could do better later
    counts = {}
    for poem in dataset:
        lang = poem["language"]
        counts[lang] = counts.get(lang, 0) + 1
    print(f"\n{name} distribution:")
    for lang, count in sorted(counts.items()):
        print(f"  {lang}: {count}")

def convert_to_bio(poem, tokenizer):
    """
    Takes a poem dict with character-offset annotations and returns
    a dict with 'tokens' and 'ner_tags' (as integer BIO label IDs).
    """
    text = poem["text"].replace("\r\n", "\n")
    annotations = ast.literal_eval(poem["responses"])["ner_tags"][0]["value"]

    # Build a character-level label map: char_idx -> (label, is_start)
    char_labels = {}
    for ann in annotations:
        if poem['language'] == 'Bulgarian':
            start = int(ann["start"])-1
            end = int(ann["start"])-1
        else:          
            start = ann["start"]
            end   = ann["end"]
        label = ann["label"]    # e.g. "PER", "LOC"
        for i in range(start, end):
            char_labels[i] = (label, i == start)

    # Tokenize with offset mapping so we know which chars each token covers
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    tokens     = encoding["input_ids"]
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

    return {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels":         bio_labels,}


def build_hf_dataset(data_split, tokenizer):
    records = []
    skipped = 0
    for poem in data_split:
        try:
            records.append(convert_to_bio(poem, tokenizer))
        except Exception as e:
            skipped += 1
            print(f"  Skipped poem {poem['id']}: {e}")
    print(f"  Converted {len(records)} poems ({skipped} skipped).")
    return Dataset.from_list(records)


def finetune(train_data, test_data, model_name, output_dir="./finetuned-ner"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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

    print("Converting train split...")
    train_ds = build_hf_dataset(train_data, tokenizer)
    print("Converting test split...")
    test_ds  = build_hf_dataset(test_data, tokenizer)

    dataset = DatasetDict({"train": train_ds, "test": test_ds})

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
        fp16=False,              # CPU  ------ we can change parameters in this part once we do HPC (like more epochs but also specify what is our configuraion somehow)
    )                             #fp16=True on HPC

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,   # fixed: was 'tokenizer'
        data_collator=data_collator,
        compute_metrics=compute_metrics_hf,
    )

    trainer.train()
    return trainer, tokenizer


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

def evaluate_finetuned(test_data, model):
    overall, label_counts, lang_stats = run_evaluation(test_data, model)
    print_results(overall, label_counts, lang_stats)

# Label schema — must match what xlm-roberta-large-finetuned-conll03 uses (I checked it in the Roberta exploration file)
LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC","I-MISC",]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


print('All functions loaded')
# CODE THAT ACTUALLY DOES SOMETHING STARTS HERE ================================================================================================================================

json_path  = "combined_annotations_final2.json"
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

#I want to further train the model on train, afterwards check the modified model on test:
train_data, test_data = stratified_split(data, train_size=0.6)
print(f"Train size (60%): {len(train_data)}")
print(f"Test size  (40%): {len(test_data)}")
print_language_distribution(train_data, "Train")
print_language_distribution(test_data, "Test")

#converting dataset into token-level BIO labels
#The model expects tokenized data with NER labels in conll or similar format - so we need to make it that way
trainer, tokenizer = finetune(train_data, test_data, model_name)
evaluate_finetuned(test_data, trainer)


# LABEL_LIST = ["O", "PER", "LOC", "ORG", "MISC"]
# label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]        #this is for BIO/IOB tagging scheme

# label2id = {label: i for i, label in enumerate(label_list)}     #Creates dictionary:{"O": 0,"B-PER": 1,"I-PER": 2,...}, 
# id2label = {i: label for label, i in label2id.items()}



#Results so far (60:40 split) ---------------------------------------------------------------------------------------------------
# All functions loaded
# Train size (60%): 238
# Test size  (40%): 159

# Train distribution:
#   Bulgarian: 60
#   Danish: 60
#   Polish: 60
#   Slovak: 58

# Test distribution:
#   Bulgarian: 41
#   Danish: 40
#   Polish: 40
#   Slovak: 38
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# [transformers] You passed `num_labels=9` which is incompatible to the `id2label` map of length `8`.
# Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 3450.63it/s]
# [transformers] XLMRobertaForTokenClassification LOAD REPORT from: FacebookAI/xlm-roberta-large-finetuned-conll03-english
# Key                         | Status     | Details                                                                                 
# ----------------------------+------------+-----------------------------------------------------------------------------------------
# roberta.pooler.dense.bias   | UNEXPECTED |                                                                                         
# roberta.pooler.dense.weight | UNEXPECTED |                                                                                         
# classifier.weight           | MISMATCH   | Reinit due to size mismatch - ckpt: torch.Size([8, 1024]) vs model:torch.Size([9, 1024])
# classifier.bias             | MISMATCH   | Reinit due to size mismatch - ckpt: torch.Size([8]) vs model:torch.Size([9])            

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISMATCH:     ckpt weights were loaded, but they did not match the original empty weight shapes.
# Converting train split...
#   Converted 238 poems (0 skipped).
# Converting test split...
#   Converted 159 poems (0 skipped).
#   0%|                                                                                                                                                                                                                                                                       | 0/357 [00:00<?, ?it/s]C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   super().__init__(loader)
# {'loss': '0.3412', 'grad_norm': '2.156', 'learning_rate': '1.894e-05', 'epoch': '0.1681'}                                                                                                                                                                                                           
# {'loss': '0.08195', 'grad_norm': '2.44', 'learning_rate': '1.782e-05', 'epoch': '0.3361'}                                                                                                                                                                                                           
# {'loss': '0.07171', 'grad_norm': '5.652', 'learning_rate': '1.669e-05', 'epoch': '0.5042'}                                                                                                                                                                                                          
# {'loss': '0.07938', 'grad_norm': '2.809', 'learning_rate': '1.557e-05', 'epoch': '0.6723'}                                                                                                                                                                                                          
# {'loss': '0.0551', 'grad_norm': '2.536', 'learning_rate': '1.445e-05', 'epoch': '0.8403'}                                                                                                                                                                                                           
# Downloading builder script: 6.34kB [00:00, 494kB/s]█████████████████████████████████████                                                                                                                                                                      | 119/357 [1:04:58<3:02:41, 46.06s/it]
# C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project\venv\Lib\site-packages\seqeval\metrics\v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.it]
#   _warn_prf(average, modifier, msg_start, len(result))
# {'eval_loss': '0.077', 'eval_precision': '0.4223', 'eval_recall': '0.5604', 'eval_f1': '0.4816', 'eval_runtime': '320.5', 'eval_samples_per_second': '0.496', 'eval_steps_per_second': '0.25', 'epoch': '1'}                                                                                        
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.56s/it]
# C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.███████████████████| 1/1 [00:08<00:00,  8.56s/it]
#   super().__init__(loader)
# {'loss': '0.08729', 'grad_norm': '14.45', 'learning_rate': '1.333e-05', 'epoch': '1.008'}                                                                                                                                                                                                           
# {'loss': '0.05379', 'grad_norm': '1.234', 'learning_rate': '1.221e-05', 'epoch': '1.176'}                                                                                                                                                                                                           
# {'loss': '0.0437', 'grad_norm': '0.7077', 'learning_rate': '1.109e-05', 'epoch': '1.345'}                                                                                                                                                                                                           
# {'loss': '0.03385', 'grad_norm': '0.8451', 'learning_rate': '9.972e-06', 'epoch': '1.513'}                                                                                                                                                                                                          
# {'loss': '0.027', 'grad_norm': '0.7365', 'learning_rate': '8.852e-06', 'epoch': '1.681'}                                                                                                                                                                                                            
# {'loss': '0.03205', 'grad_norm': '0.3338', 'learning_rate': '7.731e-06', 'epoch': '1.849'}                                                                                                                                                                                                          
#  67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                   | 238/357 [2:10:18<1:11:11, 35.90s/it]C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project\venv\Lib\site-packages\seqeval\metrics\v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior./it]
#   _warn_prf(average, modifier, msg_start, len(result))
# {'eval_loss': '0.04524', 'eval_precision': '0.6317', 'eval_recall': '0.6703', 'eval_f1': '0.6504', 'eval_runtime': '344.1', 'eval_samples_per_second': '0.462', 'eval_steps_per_second': '0.232', 'epoch': '2'}                                                                                     
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:09<00:00,  9.89s/it]
# C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.███████████████████| 1/1 [00:09<00:00,  9.89s/it]
#   super().__init__(loader)
# {'loss': '0.03175', 'grad_norm': '2.363', 'learning_rate': '6.611e-06', 'epoch': '2.017'}                                                                                                                                                                                                           
# {'loss': '0.01504', 'grad_norm': '0.1587', 'learning_rate': '5.49e-06', 'epoch': '2.185'}                                                                                                                                                                                                           
# {'loss': '0.02075', 'grad_norm': '0.2747', 'learning_rate': '4.37e-06', 'epoch': '2.353'}                                                                                                                                                                                                           
# {'loss': '0.01971', 'grad_norm': '0.52', 'learning_rate': '3.249e-06', 'epoch': '2.521'}                                                                                                                                                                                                            
# {'loss': '0.02068', 'grad_norm': '0.3193', 'learning_rate': '2.129e-06', 'epoch': '2.689'}                                                                                                                                                                                                          
# {'loss': '0.01393', 'grad_norm': '8.604', 'learning_rate': '1.008e-06', 'epoch': '2.857'}                                                                                                                                                                                                           
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [3:15:05<00:00, 20.31s/it]C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project\venv\Lib\site-packages\seqeval\metrics\v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior./it]
#   _warn_prf(average, modifier, msg_start, len(result))
# {'eval_loss': '0.04342', 'eval_precision': '0.7459', 'eval_recall': '0.7408', 'eval_f1': '0.7434', 'eval_runtime': '313.5', 'eval_samples_per_second': '0.507', 'eval_steps_per_second': '0.255', 'epoch': '3'}                                                                                     
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:13<00:00, 13.26s/it]
# {'train_runtime': '1.206e+04', 'train_samples_per_second': '0.059', 'train_steps_per_second': '0.03', 'train_loss': '0.0583', 'epoch': '3'}                                                                                                                                                         
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [3:21:03<00:00, 33.79s/it]
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.04s/it]

# Fine-tuned model saved to: ./finetuned-ner

# Loading fine-tuned model from ./finetuned-ner...
# Loading model: ./finetuned-ner ...
# Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 2651.51it/s]
# Processing of poems started.
# Processing 159/159...
# Done. There were 14 empty-annotation examples included.


# OVERALL
# ==================================================
#   Precision : 0.626
#   Recall    : 0.475
#   F1        : 0.540
#   TP=458  FP=274  FN=507


# PER LABEL
# ==================================================
#   LOC    - P: 0.616  R: 0.446  F1: 0.518 (TP=154 FP=96 FN=191)
#   MISC   - P: 0.000  R: 0.000  F1: 0.000 (TP=0 FP=3 FN=11)
#   ORG    - P: 0.000  R: 0.000  F1: 0.000 (TP=0 FP=0 FN=3)
#   PER    - P: 0.635  R: 0.502  F1: 0.560 (TP=304 FP=175 FN=302)


# PER LANGUAGE
# ==================================================
#   Bulgarian  - P: 0.061  R: 0.044  F1: 0.051 (TP=9 FP=138 FN=195)
#   Danish     - P: 0.739  R: 0.678  F1: 0.707 (TP=164 FP=58 FN=78)
#   Polish     - P: 0.734  R: 0.457  F1: 0.563 (TP=116 FP=42 FN=138)
#   Slovak     - P: 0.824  R: 0.638  F1: 0.719 (TP=169 FP=36 FN=96)
# (venv) PS C:\Users\Mirka Gemelova\OneDrive - ITU\Dokumenty\ITU\NLP\Project> 