import json
import time
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# query_result[:3]
# doc_result = embeddings.embed_documents([text])

speed_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
high_perf_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

with open('chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# chunks.json is a list of objects: [{"content": str, "metadata": {...}}, ...]
# Extract the list of text contents for embedding.
texts = [item.get('content', '') for item in data if isinstance(item, dict) and isinstance(item.get('content', ''), str) and item.get('content', '').strip()]
if not texts:
    raise ValueError("No valid 'content' strings found in chunks.json")

tic = time.time()
results = speed_embedding.embed_documents(texts)

print("Speed embedding time:", time.time() - tic)
print("Speed embedding result sample:", results)
