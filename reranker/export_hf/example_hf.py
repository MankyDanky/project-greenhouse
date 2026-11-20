# The transformers usage example from the model card, against the staged repo.
import sys, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = sys.argv[1] if len(sys.argv) > 1 else "castorini/gaggle-reranker-20261005"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

query = "what are the symptoms of iron deficiency anemia?"
passages = [
    "Iron deficiency anemia causes fatigue, shortness of breath, pale skin, and brittle nails.",
    "Iron is a metallic element with atomic number 26, widely used in construction and manufacturing.",
    "A balanced diet includes leafy greens, legumes, and red meat as sources of dietary iron.",
]
print("### OUTPUT ###")
for score, passage in model.rerank(query, passages, tokenizer=tokenizer):
    print(f"{score:+.3f}  {passage[:70]}")
