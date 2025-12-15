import os
from google import genai
from google.genai import types
import time
from pathlib import Path
from dotenv import load_dotenv

# https://blog.google/technology/developers/file-search-gemini-api/
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/File_Search.ipynb

POLL_INTERVAL = 5  # seconds

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

def run_path_finder(folder_path: str, index_name: str, output_json: str, input_text: str):
    client = genai.Client(api_key= gemini_api_key)

    file_search_store = client.file_search_stores.create(
        config=types.CreateFileSearchStoreConfig(
            display_name=index_name
        )
    )

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    files = [f for f in folder.iterdir() if f.is_file()]
    if not files:
        raise ValueError("No files found in the folder.")
    
    for file_path in files:
        upload_files = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=file_search_store.name,
            file=str(file_path),
            config=types.UploadToFileSearchStoreConfig(
                display_name= Path(file_path).name
            )
        )

        while not (upload_files := client.operations.get(upload_files)).done:
            time.sleep(2) #How many seconds to wait before polling again?

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input_text,
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store.name]
                )
            )]
        )
    )
    print(response.text)

if __name__ == "__main__":
    run_path_finder(
        folder_path="corpus",        # Folder containing files
        index_name="jadranka_index", # File Search index
        output_json="file_search_chunks.json",
        input_text="I am adventurer, recommend me what can I do in Losinj?"
        # input_text="Return ALL chunks. Do NOT summarize."
    )
