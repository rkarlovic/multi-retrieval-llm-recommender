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
    
    # Extract valid text contents
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
    
    print(f"Loaded {len(texts)} valid chunks")
    return texts, metadatas

def create_documents(texts, metadatas):
    """Create LangChain Document objects."""
    documents = []
    for text, metadata in zip(texts, metadatas):
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    return documents

def get_model_short_name(model_name):
    """Convert model name to short filename-friendly format."""
    # Extract the last part of the model name and clean it
    short_name = model_name.split('/')[-1]
    short_name = short_name.replace('-', '_').lower()
    return short_name

def create_vector_store(documents, embedding_model, model_name, base_output_dir='vector_stores'):
    """Create and save FAISS vector store with model name in path."""
    model_short_name = get_model_short_name(model_name)
    output_path = os.path.join(base_output_dir, f"vectorstore_{model_short_name}")
    
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")
    print(f"Creating vector store with {len(documents)} documents...")
    
    tic = time.time()
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    elapsed = time.time() - tic
    print(f"✓ Vector store created in {elapsed:.2f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the vector store
    vectorstore.save_local(output_path)
    print(f"✓ Vector store saved to '{output_path}/'")
    
    # Save metadata about the model
    metadata = {
        'model_name': model_name,
        'num_documents': len(documents),
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'embedding_time_seconds': elapsed
    }
    
    metadata_path = os.path.join(output_path, 'model_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Model metadata saved to '{metadata_path}'")
    
    return vectorstore, output_path

def test_vector_store(vectorstore, model_name, query="Hotel Bellevue", k=3):
    """Test the vector store with a sample query."""
    print(f"\n{'─'*60}")
    print(f"TESTING: {model_name}")
    print(f"{'─'*60}")
    print(f"Query: '{query}'")
    print(f"Retrieving top {k} results...\n")
    
    results = vectorstore.similarity_search(query, k=k)
    
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print("-" * 60)
    
    return results

def main():
    print("="*60)
    print("RAG VECTOR STORE CREATION PIPELINE")
    print("DUAL MODEL EMBEDDING")
    print("="*60)
    
    # Configuration
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
    
    CHUNKS_FILE = 'chunks.json'
    BASE_OUTPUT_DIR = 'vector_stores'
    TEST_QUERY = "luxury hotel with spa and wellness"
    
    # Load chunks once (used for both models)
    texts, metadatas = load_chunks(CHUNKS_FILE)
    
    # Create LangChain documents
    print("\nCreating LangChain Document objects...")
    documents = create_documents(texts, metadatas)
    print(f"✓ Created {len(documents)} documents")
    
    # Process each model
    results_summary = []
    
    for model_config in MODELS:
        model_name = model_config['name']
        
        try:
            # Initialize embedding model
            print(f"\n{'='*60}")
            print(f"Initializing: {model_name}")
            print(f"Description: {model_config['description']}")
            print(f"{'='*60}")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': model_config['device']},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✓ Embedding model loaded successfully")
            
            # Create and save vector store
            vectorstore, output_path = create_vector_store(
                documents, 
                embeddings, 
                model_name,
                BASE_OUTPUT_DIR
            )
            
            # Test the vector store
            test_vector_store(vectorstore, model_name, query=TEST_QUERY, k=3)
            
            results_summary.append({
                'model': model_name,
                'status': 'SUCCESS',
                'path': output_path
            })
            
        except Exception as e:
            print(f"\n✗ ERROR processing {model_name}: {str(e)}")
            results_summary.append({
                'model': model_name,
                'status': 'FAILED',
                'error': str(e)
            })
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print("\nSUMMARY:")
    print("-" * 60)
    
    for result in results_summary:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{status_symbol} {result['model']}")
        if result['status'] == 'SUCCESS':
            print(f"  Location: {result['path']}/")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        print()
    
    print("To load a vector store later, use:")
    print("\n  from langchain_huggingface.embeddings import HuggingFaceEmbeddings")
    print("  from langchain_community.vectorstores import FAISS")
    print("\n  # For MiniLM model:")
    print("  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')")
    print("  vectorstore = FAISS.load_local('vector_stores/vectorstore_all_minilm_l6_v2', embeddings)")
    print("\n  # For BGE-M3 model:")
    print("  embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')")
    print("  vectorstore = FAISS.load_local('vector_stores/vectorstore_bge_m3', embeddings)")
    print("\n  # Perform search:")
    print("  results = vectorstore.similarity_search('your query', k=5)")

if __name__ == "__main__":
    main()