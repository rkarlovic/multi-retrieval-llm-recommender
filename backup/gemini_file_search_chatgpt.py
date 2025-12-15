import os
import time
import json
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

POLL_INTERVAL = 5  # seconds

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


def create_or_get_store(client, store_name):
    """Create a File Search store (or reuse one)."""
    try:
        # Try to get existing store
        store = client.file_search_stores.get(name=store_name)
        print(f"Store already exists: {store_name}")
        return store
    except Exception:
        print(f"Creating new File Search store: {store_name}")
        return client.file_search_stores.create(
            config={"display_name": store_name}
        )


def upload_all_files(client, folder_path, store_name):
    """Upload all files to the File Search store."""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    files = [str(f) for f in folder.iterdir() if f.is_file()]
    if not files:
        raise ValueError("No files found in the folder.")

    print(f"\n📤 Uploading files to store '{store_name}'...")

    for file_path in files:
        print(f" → Uploading {file_path}")

        operation = client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config={"display_name": Path(file_path).name}
        )

        # Poll import operation
        while not operation.done:
            print("   ...processing...")
            time.sleep(POLL_INTERVAL)
            operation = client.operations.get(operation)

    print(f"✔ Uploaded {len(files)} files.")


def export_chunks(client, store_name, output_file):
    """Query store and save chunks to local JSON."""
    print("\n📥 Exporting chunks...")

    # FIXED: Correct syntax for file_search tool
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Return ALL chunks. Do NOT summarize.",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ]
        )
    )
    
    print(f"\n📝 Response text:\n{response.text}\n")

    # Access grounding metadata
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            grounding_metadata = candidate.grounding_metadata
            
            if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
                chunks = []
                for chunk in grounding_metadata.grounding_chunks:
                    chunk_data = {
                        "text": None,
                        "title": None,
                        "uri": None,
                    }
                    
                    if hasattr(chunk, 'retrieved_context'):
                        if hasattr(chunk.retrieved_context, 'text'):
                            chunk_data["text"] = chunk.retrieved_context.text
                        if hasattr(chunk.retrieved_context, 'title'):
                            chunk_data["title"] = chunk.retrieved_context.title
                        if hasattr(chunk.retrieved_context, 'uri'):
                            chunk_data["uri"] = chunk.retrieved_context.uri
                    
                    chunks.append(chunk_data)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)

                print(f"✔ Saved {len(chunks)} chunks → {output_file}")
                return

    print("⚠️ No grounding chunks returned. This may mean:")
    print("  - The store is still processing")
    print("  - The query didn't match any content")
    print("  - The files haven't been indexed yet")


def run_full_pipeline(folder_path, store_name, output_json):
    client = genai.Client(api_key=gemini_api_key)

    print("\n🚀 Running Gemini File Search pipeline (SDK 1.52.0)...")

    store = create_or_get_store(client, store_name)
    # Use store.name (the auto-generated ID) instead of display_name
    upload_all_files(client, folder_path, store.name)
    export_chunks(client, store.name, output_json)

    print("\n🎉 DONE!")


if __name__ == "__main__":
    run_full_pipeline(
        folder_path="corpus",
        store_name="projects/350005989094",
        output_json="file_search_chunks.json"
    )