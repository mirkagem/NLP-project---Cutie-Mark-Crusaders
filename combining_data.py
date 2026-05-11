import json
import csv
import sys
import glob
from pathlib import Path
import re

languages = {"BG": "Bulgarian","SK": "Slovak","DK": "Danish","PL": "Polish"}

def get_language(filepath: Path) -> str:
    stem = filepath.stem                    
    code = stem.split("_")[-1].upper()     
    return languages.get(code, code)     

def load_json(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{filepath}: expected a JSON array or object, got {type(data)}")
    for record in data:
        if isinstance(record.get("text"), dict):
            record["text"] = record["text"].get("text", "")
    return data

def combine(filepaths: list[Path]) -> list[dict]:
    combined = []
    for fp in filepaths:
        lang = get_language(fp)
        records = load_json(fp)
        for record in records:
            record["language"] = lang
            combined.append(record)
        print(f"  {fp.name}: {len(records)} records  (language='{lang}')")
    return combined

def save_json(data: list[dict], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    if len(sys.argv) > 1:
        filepaths = [Path(p) for arg in sys.argv[1:] for p in glob.glob(arg)]
    else:
        filepaths = sorted(Path(".").glob("*.json"))
    filepaths = [fp for fp in filepaths if fp.is_file()]
    if not filepaths:
        print("No JSON files found. Pass file paths as arguments or run in a folder with .json files.")
        sys.exit(1)

    print(f"Combining {len(filepaths)} file(s)...")
    combined = combine(filepaths)
    print(f"Total records: {len(combined)}\n")

    out_json = Path("combined_annotations_final.json")

    save_json(combined, out_json)
    print(f"Saved JSON → {out_json}")

if __name__ == "__main__":
    main()
    