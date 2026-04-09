import argilla as rg
import pandas as pd

file_name = input("Enter the filename (e.g., data.txt): ")

client = rg.Argilla(api_url="http://localhost:6900", api_key="admin.apikey")

settings = rg.Settings(
    fields=[
        rg.TextField(name="text", title="Raw Text")
    ],
    questions=[
        rg.SpanQuestion(
            name="ner_tags",
            field="text",
            labels=["PER", "LOC", "ORG", "MISC"], # Change these to your project labels
            title="Identify Entities"
        )
    ],
    guidelines="Highlight the name, location, or organization in the text."
)

dataset = rg.Dataset(
    name="my_ner_project",
    settings=settings,
    workspace="admin"
)
dataset.create()

df = pd.read_csv(file_name)

records = [rg.Record(fields={"text": str(row["sentence"])}) for _, row in df.iterrows()]
dataset.records.log(records)

print("Dataset created! Refresh your browser.")