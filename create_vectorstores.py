"""
One-time script to create FAISS vector stores from chunks.json.

This creates vector stores for multiple embedding models that can later be loaded
by the EmbeddingRetriever class.
"""
import json
import time
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def load_chunks(filepath='chunks.json'):
    """Load chunks from JSON file."""
    print(f"Loading chunks from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    metadatas = []
    
    for item in data:
        if isinstance(item, dict) and isinstance(item.get('content', ''), str):
            content = item.get('content', '').strip()
            if content:
                texts.append(content)
                metadatas.append(item.get('metadata', {}))
    
    if not texts:
        raise ValueError("No valid 'content' strings found in chunks.json")
    
    print(f"✓ Loaded {len(texts)} valid chunks")
    return texts, metadatas


def create_documents(texts, metadatas):
    """Create LangChain Document objects."""
    documents = [Document(page_content=text, metadata=metadata) 
                 for text, metadata in zip(texts, metadatas)]
    return documents


def get_model_short_name(model_name):
    """Convert model name to short filename-friendly format."""
    short_name = model_name.split('/')[-1]
    return short_name.replace('-', '_').lower()


def create_vector_store(documents, embedding_model, model_name, base_output_dir='vector_stores'):
    """Create and save FAISS vector store."""
    model_short_name = get_model_short_name(model_name)
    output_path = os.path.join(base_output_dir, f"vectorstore_{model_short_name}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    tic = time.time()
    vectorstore = FAISS.from_documents(documents, embedding_model)
    elapsed = time.time() - tic
    
    print(f"✓ Vector store created in {elapsed:.2f} seconds")
    
    os.makedirs(output_path, exist_ok=True)
    vectorstore.save_local(output_path)
    print(f"✓ Saved to '{output_path}/'")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'num_documents': len(documents),
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'embedding_time_seconds': elapsed
    }
    
    metadata_path = os.path.join(output_path, 'model_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return vectorstore, output_path


def main():
    print("="*60)
    print("FAISS VECTOR STORE CREATION")
    print("="*60)
    
    # Models to create vector stores for
    MODELS = [
        {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'description': 'Fast and efficient (384 dimensions)',
            'device': 'cpu'
        },
        {
            'name': 'BAAI/bge-m3',
            'description': 'High quality multilingual (1024 dimensions)',
            'device': 'cpu'  # Change to 'cuda' if GPU available
        }
    ]
    
    # Load chunks once
    texts, metadatas = load_chunks('chunks.json')
    documents = create_documents(texts, metadatas)
    print(f"✓ Created {len(documents)} documents\n")
    
    # Process each model
    results = []
    
    for model_config in MODELS:
        model_name = model_config['name']
        
        try:
            print(f"Initializing: {model_name}")
            print(f"Description: {model_config['description']}")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': model_config['device']},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            vectorstore, output_path = create_vector_store(
                documents, embeddings, model_name
            )
            
            results.append({
                'model': model_name,
                'status': 'SUCCESS',
                'path': output_path
            })
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            results.append({
                'model': model_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    for result in results:
        status = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{status} {result['model']}")
        if result['status'] == 'SUCCESS':
            print(f"  → {result['path']}/")
        else:
            print(f"  → Error: {result.get('error', 'Unknown')}")
        print()


if __name__ == "__main__":
    main()
