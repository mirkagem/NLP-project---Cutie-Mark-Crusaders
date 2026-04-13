#pip install transformers torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple" #Outputs [{'entity_group': 'ORG', 'score': np.float32(0.9950118), 'word': 'Obecným Úradom', 'start': 5, 'end': 19}, {'entity_group': 'LOC', 'score': np.float32(0.99993354), 'word': 'Detve', 'start': 22, 'end': 27}]
)

text = 'Moja mama Inga Bohatá má Emu v žemli z Budimíra.' #this should be changed to be text from our not annotated file (for which we have the gold standard)

results = ner(text)
print(results)