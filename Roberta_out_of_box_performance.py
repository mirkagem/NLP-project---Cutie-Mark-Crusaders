import json
import ast
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

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

def verify_offsets(data, num_examples=10, language=None):
    checked = 0
    for poem in data:
        if language and poem["language"] != language:   #only check language which I want to check (unless no language given)
            continue
        annotations = ast.literal_eval(poem["responses"])["ner_tags"][0]["value"]   #from the string(which it originally is), make it a real dictionary + take just what is in value
                                                                                    #result would be stg like: [{'label': 'PER', 'start': 1395, 'end': 1403}, {'label': 'LOC', 'start': 1595, 'end': 1603}] 
        if not annotations:    #to not print annotations without entities
            continue
        text = poem["text"].replace("\r\n", "\n")
 
        print(f"\nID: {poem['id']}")
        print(f"Language: {poem['language']}")
        for annot in annotations:
            #adjusting Bulgarian offset
            if poem["language"] == "Bulgarian":
                start = int(annot["start"]) - 1
                end = int(annot["end"]) - 1
            else:
                start = annot["start"]
                end = annot["end"]

            extracted_entity = text[start:end]
            print(f"  Label: {annot['label']} | [{annot['start']}:{annot['end']}] | '{extracted_entity}'")
        checked += 1
        if checked >= num_examples:
            break

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
            gold_set = spans_to_set(annotations, text, language)
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

# CODE THAT ACTUALLY DOES SOMETHING STARTS HERE ================================================================================================================================

json_path  = "combined_annotations_final2.json"
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

#checking offset 
print("\n--- Offset Verification START ---")
# verify_offsets(data, num_examples=100, language="Bulgarian")      #Slovak, Danish work well, Polish works in general except 5 poems (to be Deleted), Bulgarian is same case as Polish
# input("Check the output above. If the extracted text looks correct, press Enter to continue...")
print("\n--- Offset Verification DONE ---\n")

#running the model + evaluation
ner_pipeline = load_ner_pipeline(model_name)
overall, label_counts, lang_counts = run_evaluation(data, ner_pipeline)
print_results(overall, label_counts, lang_counts)







# SAVED RESULTS SO FAR ------------------------------------------------------------------------------------------



#Result 1 (using original final merged dataset):
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


#New final (removed faulty formatings - combined_annotations_final2.json with 389 poems only):
# Done. Skipped 36 empty-annotation examples.

# ==================================================
# OVERALL
# ==================================================
#   Precision : 0.557
#   Recall    : 0.444
#   F1        : 0.494
#   TP=964  FP=766  FN=1208

# ==================================================
# PER LABEL
# ==================================================
#   LOC    → P: 0.681  R: 0.592  F1: 0.634  (TP=455 FP=213 FN=313)
#   MISC   → P: 0.013  R: 0.148  F1: 0.024  (TP=4 FP=300 FN=23)
#   ORG    → P: 0.182  R: 0.333  F1: 0.235  (TP=2 FP=9 FN=4)
#   PER    → P: 0.673  R: 0.367  F1: 0.475  (TP=503 FP=244 FN=868)

# ==================================================
# PER LANGUAGE
# ==================================================
#   Bulgarian  → P: 0.755  R: 0.738  F1: 0.747  (TP=324 FP=105 FN=115)
#   Danish     → P: 0.509  R: 0.452  F1: 0.479  (TP=270 FP=260 FN=327)
#   Polish     → P: 0.406  R: 0.262  F1: 0.318  (TP=143 FP=209 FN=403)
#   Slovak     → P: 0.542  R: 0.385  F1: 0.450  (TP=227 FP=192 FN=363)


#All of the above excluded poems with no entities, here it is with including them:
# Processing 397/397...
# Done. There were 36 empty-annotation examples included.


# OVERALL
# ==================================================
#   Precision : 0.539
#   Recall    : 0.430
#   F1        : 0.478
#   TP=960  FP=821  FN=1273

# PER LABEL
# ==================================================
#   LOC    - P: 0.664  R: 0.578  F1: 0.618 (TP=457 FP=231 FN=334)
#   MISC   - P: 0.012  R: 0.148  F1: 0.023 (TP=4 FP=318 FN=23)
#   ORG    - P: 0.167  R: 0.333  F1: 0.222 (TP=2 FP=10 FN=4)
#   PER    - P: 0.655  R: 0.353  F1: 0.458 (TP=497 FP=262 FN=912)


# PER LANGUAGE
# ==================================================
#   Bulgarian  - P: 0.723  R: 0.706  F1: 0.714 (TP=310 FP=119 FN=129)
#   Danish     - P: 0.507  R: 0.452  F1: 0.478 (TP=271 FP=264 FN=328)
#   Polish     - P: 0.378  R: 0.245  F1: 0.297 (TP=138 FP=227 FN=426)
#   Slovak     - P: 0.533  R: 0.382  F1: 0.445 (TP=241 FP=211 FN=390)
