import json
json_path  = "combined_annotations_final2.json"
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"

with open("combined_annotations_final.json", encoding="utf-8") as f:
    data = json.load(f)

#make new json file
l = []
limit_SK = 10
limit_DK = 10
for poem in data:
    
    if poem['language'] == 'Slovak' and limit_SK > 0:
        limit_SK = limit_SK - 1
        l.append(poem)
    elif poem['language'] == 'Danish' and limit_DK > 0:
        limit_DK = limit_DK - 1
        l.append(poem)

# put everything from the list to the new json file:
with open('testing_json.json', 'w', encoding='utf-8') as f:
    json.dump(l, f, ensure_ascii=False, indent=2)