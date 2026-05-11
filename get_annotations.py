import argilla as rg
import json

annotation_name = input("Enter the name of your annotation(e.g., Final annotation): ")

client = rg.Argilla(
    api_url="http://localhost:6900",
    api_key="admin.apikey"
)

dataset = client.datasets("annotation_name")

records = list(dataset.records())

data = [
    {
        "id": r.id,
        "text": r.fields,
        "responses": str(r.responses)
    }
    for r in records
]

with open("annotations.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)