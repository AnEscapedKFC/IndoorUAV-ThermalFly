"""Count literal English words/phrases in the authors' released task metadata."""
import json
from pathlib import Path
import re
import zipfile


here = Path(__file__).resolve().parent
path = here.parents[1] / "upstream/IndoorUAV-Agent/training_data.zip"
member = "training_data/training_data/meta/tasks.jsonl"
terms = ["forward", "backward", "backwards", "back", "turn left", "turn right", "sideways",
         "move left", "move right", "upward", "downward", "ascend", "descend", "hover",
         "stop", "slow", "slowly", "accelerate", "speed", "in place"]
with zipfile.ZipFile(path) as archive:
    tasks = [json.loads(line)["task"] for line in archive.open(member)]
document = {"source": str(path), "member": member, "task_rows": len(tasks), "unique_texts": len(set(tasks)),
            "unit": "Distinct task-index rows containing at least one case-insensitive whole-word/phrase match",
            "counts": {}, "rare_examples": {}}
for term in terms:
    pattern = re.compile(r"\b" + re.escape(term) + r"\b", flags=re.IGNORECASE)
    matched = [text for text in tasks if pattern.search(text)]
    document["counts"][term] = len(matched)
    if len(matched) <= 3:
        document["rare_examples"][term] = matched
(here / "training_word_counts.json").write_text(json.dumps(document, indent=2) + "\n")
