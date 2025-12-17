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

tic = time.time()
results = speed_embedding.embed_documents(data['content'])

print("Speed embedding time:", time.time() - tic)
print("Speed embedding result sample:", results)
