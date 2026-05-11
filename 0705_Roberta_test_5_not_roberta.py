#THIS IS VERSION 4 = I found out in Slovak it messes up the offset for poems with /r or /r/n in them (/n alone works fine)


#version 5 should remove commans and interpunction at the end of an entity

import json
import ast
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

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
        aggregation_strategy="simple")


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


def verify_offsets(data, num_examples=3, language=None):
    print("\n--- OFFSET VERIFICATION ---")
    checked = 0
    for item in data:
        if language and item["language"] != language:
            continue
        annotations = parse_annotations(item["responses"])
        if not annotations:
            continue
        text = extract_text(item)       # ← use extract_text here
        print(f"\nID: {item['id']}")
        print(f"Language: {item['language']}")
        for ann in annotations:
            extracted = get_span(text, ann["start"], ann["end"], item["language"])
            print(f"  Label: {ann['label']} | [{ann['start']}:{ann['end']}] | '{extracted}'")
        checked += 1
        if checked >= num_examples:
            break
    print("\n--- END VERIFICATION ---\n")


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


#ACTUALLY RUNNING THE CODE: --------------------------------------------------------------------------------------

# if __name__ == "__main__":
JSON_PATH  = "combined_annotations_final.json"      #changed in version 4
MODEL_NAME = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"   #Davlan/distilbert-base-multilingual-cased-ner-hrl)

data = load_data(JSON_PATH)
ner_pipeline = load_ner_pipeline(MODEL_NAME)

# STEP A: verify offsets look correct before full run
# verify_offsets(data, num_examples=200, language="Bulgarian") #Slovak, Danish work well, Polish works in general except 5 poems (to be Deleted), Bulgarian is same case as Polish
# input("Check the output above. If the extracted text looks correct, press Enter to continue...")

# # STEP B: run full evaluation
overall, label_counts, lang_counts = run_evaluation(data, ner_pipeline)
print_results(overall, label_counts, lang_counts)



#-------------------------- Result 1 (using original final merged dataset, ROberta):
# ==================================================
# OVERALL
# ==================================================
#   Precision : 0.541
#   Recall    : 0.430
#   F1        : 0.479
#   TP=960  FP=816  FN=1273

# ==================================================
# PER LABEL
# ==================================================
#   LOC    → P: 0.664  R: 0.578  F1: 0.618  (TP=457 FP=231 FN=334)
#   MISC   → P: 0.013  R: 0.148  F1: 0.023  (TP=4 FP=314 FN=23)
#   ORG    → P: 0.167  R: 0.333  F1: 0.222  (TP=2 FP=10 FN=4)
#   PER    → P: 0.656  R: 0.353  F1: 0.459  (TP=497 FP=261 FN=912)

# ==================================================
# PER LANGUAGE
# ==================================================
#   Bulgarian  → P: 0.723  R: 0.706  F1: 0.714  (TP=310 FP=119 FN=129)
#   Danish     → P: 0.507  R: 0.452  F1: 0.478  (TP=271 FP=263 FN=328)
#   Polish     → P: 0.380  R: 0.245  F1: 0.298  (TP=138 FP=225 FN=426)
#   Slovak     → P: 0.536  R: 0.382  F1: 0.446  (TP=241 FP=209 FN=390)

# ---------------------- Result 2: (using original final merged dataset, Davlan model (Davlan/distilbert-base-multilingual-cased-ner-hrl))
# ==================================================
# OVERALL
# ==================================================
#   Precision : 0.296
#   Recall    : 0.318
#   F1        : 0.307
#   TP=711  FP=1694  FN=1522

# ==================================================
# PER LABEL
# ==================================================
#   LOC    → P: 0.255  R: 0.392  F1: 0.309  (TP=310 FP=908 FN=481)
#   MISC   → P: 0.000  R: 0.000  F1: 0.000  (TP=0 FP=0 FN=27)
#   ORG    → P: 0.050  R: 0.333  F1: 0.087  (TP=2 FP=38 FN=4)
#   PER    → P: 0.348  R: 0.283  F1: 0.312  (TP=399 FP=748 FN=1010)

# ==================================================
# PER LANGUAGE
# ==================================================
#   Bulgarian  → P: 0.651  R: 0.595  F1: 0.621  (TP=261 FP=140 FN=178)
#   Danish     → P: 0.161  R: 0.337  F1: 0.218  (TP=202 FP=1056 FN=397)
#   Polish     → P: 0.283  R: 0.193  F1: 0.230  (TP=109 FP=276 FN=455)
#   Slovak     → P: 0.385  R: 0.220  F1: 0.280  (TP=139 FP=222 FN=492)


#Things we could tweak:
#aggregation strategy in def load_ner_pipeline: 
    # aggregation_strategy="simple"    # current
    # aggregation_strategy="first"     # uses first token's score for the word
    # aggregation_strategy="average"   # averages scores across tokens
    # aggregation_strategy="max"       # uses highest scoring token
#filtering out low-confidence predictions in def predictions_to_set:
    #def predictions_to_set(predictions, threshold=0.5):
        # return {(p["start"], p["end"], p["entity_group"]) for p in predictions if p["score"] >= threshold}
    #Expected performance: Higher threshold = higher precision but lower recall.
#trying different models
    # Davlan/distilbert-base-multilingual-cased-ner-hrl     ------