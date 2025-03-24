import requests
import json
import os
import time
from tqdm import tqdm

os.makedirs("recordings", exist_ok=True)

base_url = "https://www.xeno-canto.org/api/2/recordings?query=q:A+B"

page = 1
downloaded = 0
max_downloads = 3000  

while downloaded < max_downloads:
    print(f"fetching page {page}...")

    response = requests.get(f"{base_url}&page={page}")
    data = response.json()

    if "recordings" not in data or not data["recordings"]:
        print("no more recordings found")
        break

    for rec in tqdm(data["recordings"]):
        if downloaded >= max_downloads:
            break 

        file_url = rec['file'] 
        filename = f"recordings/{rec['id']}.wav"

        try:
            with open(filename, "wb") as f:
                f.write(requests.get(file_url).content)
            downloaded += 1
        except Exception as e:
            print(f"failed to download {file_url}: {e}")

    if page >= data["numPages"]:
        print("reached last available page")
        break

    page += 1
    time.sleep(1)

print(f"finished! {downloaded} files downloaded.")
