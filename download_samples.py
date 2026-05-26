import os
import urllib.request
import zipfile

def download_file(url, save_path):
    print(f"Downloading {url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Saved to {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise e

def main():
    zip_url = "https://archive.ics.uci.edu/static/public/475/rice+leaf+diseases.zip"
    zip_path = "rice_leaf_diseases.zip"
    extract_dir = "dataset"
    
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        download_file(zip_url, zip_path)
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Successfully extracted to {extract_dir}")
        
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up zip file.")
    except Exception as e:
        print(f"Failed to process dataset: {e}")

if __name__ == "__main__":
    main()
