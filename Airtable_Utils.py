import os
import requests

# ✏️ CONFIGURATION
AIRTABLE_API_KEY = 'patU7WPdqAafxgaIz.3c314b7994bbad26c573cdcbdc8dfc63448ba92e87a7753c6e96a44d11e1f199'  # or bearer token
BASE_ID = 'appdIO1tZxoyAvleh'
TABLE_NAME = 'Chinook Project Test Results'
ATTACHMENT_FIELD = 'RAW data attachment'
DOWNLOAD_DIR = './downloads'

# Airtable API endpoint
url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}'

# Headers
headers = {
    'Authorization': f'Bearer {AIRTABLE_API_KEY}'
}

# Create directory if not exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Track number of files downloaded and duplicates
download_count = 0
duplicate_files = []

# Pagination support
offset = None
while True:
    params = {}
    if offset:
        params['offset'] = offset

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    for record in data['records']:
        attachments = record.get('fields', {}).get(ATTACHMENT_FIELD, [])
        for attachment in attachments:
            file_url = attachment['url']
            file_name = attachment.get('filename', file_url.split('/')[-1])
            # Only download CSV files
            if file_name.lower().endswith('.csv'):
                try:
                    print(f'Downloading {file_name}...')
                    file_data = requests.get(file_url)
                    file_data.raise_for_status()  # Check for HTTP errors
                    
                    # Add timestamp or record ID to prevent overwriting files with same name
                    output_path = os.path.join(DOWNLOAD_DIR, file_name)
                    if os.path.exists(output_path):
                        duplicate_files.append(file_name)
                        name, ext = os.path.splitext(file_name)
                        output_path = os.path.join(DOWNLOAD_DIR, f"{name}_{record['id']}{ext}")
                        print(f"File already exists, saving as {output_path}")
                    
                    with open(output_path, 'wb') as f:
                        f.write(file_data.content)
                    download_count += 1
                except Exception as e:
                    print(f"Error downloading {file_name}: {e}")
            else:
                print(f'Skipping non-CSV file: {file_name}')

    # Continue if more records
    offset = data.get('offset')
    if not offset:
        break

print(f'✅ All attachments downloaded. Total CSV files: {download_count}')
if duplicate_files:
    print(f"\nFound {len(duplicate_files)} duplicate filenames:")
    for file in duplicate_files:
        print(f"  - {file}")
