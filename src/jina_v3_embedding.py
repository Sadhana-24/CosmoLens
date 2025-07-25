import requests
import json
import os
from datasets import load_dataset
import time

# --- Configuration ---
DATASET_NAME = "UniverseTBD/AstroLLaVA_convos"
DATASET_SPLIT = "train"
TEXT_COLUMN = "conversation"
MAX_ROWS = 10000  # Process only the first 100 rows
BATCH_SIZE = 50  # Smaller batch size for more frequent updates
OUTPUT_FILE = "astronomy_embeddings_100.jsonl" # Changed extension to .jsonl
JINA_API_URL = 'https://api.jina.ai/v1/embeddings'
JINA_MODEL = "jina-embeddings-v3"

# --- API Keys ---
JINA_API_KEYS = [
    'jina_bbc0a956bcd74f3e83b71a49c32f1f6c2ct1puE5gnTNZzFkPwmWP0phEofH',
    # Add more keys here if you have them
]
current_key_index = 0

# --- Helper Function to Convert Conversation Dict to String ---
def conversation_to_string(conversation_dict):
    """Convert the conversation dictionary to a single string for embedding"""
    if not isinstance(conversation_dict, dict):
        return ""

    conversation_text = ""

    # Extract all messages from the conversation dictionary
    # Check if 'conversation' key exists and is a list
    if 'conversation' in conversation_dict and isinstance(conversation_dict['conversation'], list):
        for msg_data in conversation_dict['conversation']:
             # Check if msg_data is a dict and has 'value'
             if isinstance(msg_data, dict) and 'value' in msg_data and isinstance(msg_data['value'], str):
                 # Add space between messages if needed
                 if conversation_text:
                     conversation_text += " "
                 conversation_text += msg_data['value']
    # Fallback or alternative structure check (like the original 'from'/'value' lists)
    elif all(role in conversation_dict and isinstance(conversation_dict[role], list) for role in ["from", "value"]):
         # Assuming 'value' list contains the text parts
         for i, msg in enumerate(conversation_dict.get("value", [])):
             if isinstance(msg, str):
                 if conversation_text: # Add space only if text already exists
                     conversation_text += " "
                 conversation_text += msg

    return conversation_text

# --- Helper Function for API Call with Key Rotation ---
def get_jina_embeddings(texts_batch):
    """Sends a batch of texts to Jina API and handles key rotation."""
    global current_key_index
    max_retries = len(JINA_API_KEYS)

    for attempt in range(max_retries):
        if not JINA_API_KEYS:
            print("Error: No JINA API keys provided.")
            return None
        api_key = JINA_API_KEYS[current_key_index]
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        # Ensure texts_batch contains non-empty strings
        valid_texts_batch = [text for text in texts_batch if isinstance(text, str) and text.strip()]
        if not valid_texts_batch:
             print("Warning: Skipping batch because it contains no valid text.")
             return [] # Return empty list as no embeddings were generated

        data = {
            "model": JINA_MODEL,
            "input": [{"text": text} for text in valid_texts_batch] # Use the filtered batch
        }

        try:
            response = requests.post(JINA_API_URL, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()

            response_data = response.json()
            # Check for quota error specifically
            if isinstance(response_data, dict) and response_data.get("detail") and "quota" in str(response_data["detail"]).lower():
                print(f"Quota likely exceeded for key index {current_key_index}. Trying next key.")
                current_key_index = (current_key_index + 1) % len(JINA_API_KEYS)
                continue

            if 'data' in response_data and isinstance(response_data['data'], list):
                # Ensure we only extract embeddings if they exist
                embeddings = [item['embedding'] for item in response_data['data'] if 'embedding' in item]
                # Check if the number of embeddings matches the number of valid texts sent
                if len(embeddings) == len(valid_texts_batch):
                    return embeddings
                else:
                    print(f"Warning: Mismatch in expected ({len(valid_texts_batch)}) and received ({len(embeddings)}) embeddings.")
                    # Decide how to handle mismatch, e.g., return partial or None
                    # For now, let's try the next key as it might be a partial success/error
                    current_key_index = (current_key_index + 1) % len(JINA_API_KEYS)
                    continue

            else:
                print(f"Unexpected response format from Jina API: {response_data}")
                current_key_index = (current_key_index + 1) % len(JINA_API_KEYS)
                continue

        except requests.exceptions.RequestException as e:
            print(f"Request failed for key index {current_key_index}: {e}")
            # Handle rate limits (429) or other specific errors
            status_code = None
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                # Check for quota exceeded via status code if applicable (e.g., 402 Payment Required)
                if status_code == 402: # Example status code for quota
                     print(f"Quota exceeded (HTTP {status_code}) for key index {current_key_index}. Trying next key.")
                elif status_code == 429:
                    print("Rate limit hit. Waiting and trying next key...")
                    time.sleep(5) # Wait before retrying with the next key
                else:
                     print(f"Received HTTP error {status_code}.")

            current_key_index = (current_key_index + 1) % len(JINA_API_KEYS)
            # Optional: Add a small delay before trying the next key even for non-429 errors
            # time.sleep(1)

            if attempt == max_retries - 1:
                print("All API keys failed or encountered errors for this batch.")
                return None # Failed to get embeddings for the batch

    print("Failed to get embeddings after trying all keys for this batch.")
    return None

# --- Main Processing Logic ---
print(f"Loading dataset {DATASET_NAME}, split {DATASET_SPLIT} in streaming mode...")
try:
    # Load dataset in streaming mode
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Remove the output file if it exists to start fresh
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"Removed existing output file: {OUTPUT_FILE}")

total_processed_count = 0
current_batch_texts = []
current_batch_metadata = []

print(f"Starting embedding process for first {MAX_ROWS} rows, saving to {OUTPUT_FILE}...")

# Process the dataset in streaming mode
try: # Wrap the loop in try...except to catch potential dataset iteration errors
    for row_idx, row in enumerate(dataset):
        if row_idx >= MAX_ROWS:
            break

        # Safely access 'conversation' column
        conversation_data = row.get(TEXT_COLUMN)
        if conversation_data is None:
             print(f"Skipping row {row_idx} - '{TEXT_COLUMN}' column missing or None.")
             continue

        conversation_text = conversation_to_string(conversation_data)

        # Skip empty conversations
        if not conversation_text or not conversation_text.strip():
            print(f"Skipping row {row_idx} - empty or invalid conversation content")
            continue

        # Prepare metadata - ensure keys exist in the row or handle missing keys
        metadata = {
            "id": row.get("id", f"missing_id_{row_idx}"), # Provide default if 'id' is missing
            "caption": row.get("caption", ""), # Default to empty string if missing
            "url": row.get("url", ""),       # Default to empty string if missing
            "conversation_text": conversation_text, # Already processed text
            "row_index": row_idx
        }

        # Add to current batch
        current_batch_texts.append(conversation_text)
        current_batch_metadata.append(metadata)

        # Process when batch is full or we've reached the end (and have items in batch)
        if (len(current_batch_texts) >= BATCH_SIZE or row_idx == MAX_ROWS - 1) and current_batch_texts:
            print(f"Processing batch of {len(current_batch_texts)} texts (Row index up to: {row_idx})...")

            embeddings = get_jina_embeddings(current_batch_texts)

            if embeddings is not None: # Check for None explicitly (means total failure for batch)
                 if len(embeddings) == len(current_batch_texts):
                     # Save this batch to JSONL file immediately
                     try:
                         with open(OUTPUT_FILE, 'a', encoding='utf-8') as f: # Open in append mode
                             for j, embedding in enumerate(embeddings):
                                 record = {
                                     "embedding": embedding,
                                     "metadata": current_batch_metadata[j]
                                 }
                                 f.write(json.dumps(record) + '\n') # Write as JSON line
                         print(f"Successfully processed and saved batch ending at row {row_idx}. Total saved: {total_processed_count + len(embeddings)}")
                         total_processed_count += len(embeddings)
                     except IOError as e:
                         print(f"Error writing batch to file {OUTPUT_FILE}: {e}")
                         # Decide if you want to stop or continue
                         # break # Example: stop if writing fails
                 elif len(embeddings) > 0: # Handle partial success if get_jina_embeddings could return partial lists
                      print(f"Warning: Processed batch partially. Expected {len(current_batch_texts)}, got {len(embeddings)} embeddings. Saving received embeddings.")
                      # Code to handle partial results (e.g., match based on order, if reliable)
                      # This part is tricky and depends on how partial results are returned.
                      # Assuming order is preserved for the successful ones:
                      try:
                          with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                              for j in range(len(embeddings)): # Only iterate up to the number of embeddings received
                                   record = {
                                       "embedding": embeddings[j],
                                       "metadata": current_batch_metadata[j] # Assumes metadata order matches
                                   }
                                   f.write(json.dumps(record) + '\n')
                          print(f"Successfully saved partial batch ({len(embeddings)} items) ending at row {row_idx}. Total saved: {total_processed_count + len(embeddings)}")
                          total_processed_count += len(embeddings)
                      except IOError as e:
                          print(f"Error writing partial batch to file {OUTPUT_FILE}: {e}")

                 else: # len(embeddings) == 0 but not None (e.g., empty valid batch sent)
                      print(f"Batch ending at row {row_idx} resulted in zero embeddings (e.g., all input texts were invalid).")

            else:
                # get_jina_embeddings returned None, indicating failure for the whole batch
                print(f"Warning: Failed to get embeddings for batch ending at row {row_idx} after trying all keys. Skipping this batch.")
                # Optionally add failed metadata/texts to a separate error log

            # Clear the batch regardless of success or failure for this attempt
            current_batch_texts = []
            current_batch_metadata = []

except StopIteration:
     print("Finished iterating through the dataset stream.")
except Exception as e:
     print(f"An error occurred during dataset processing loop: {e}")
     # Consider saving any remaining items in the current batch if needed
     # if current_batch_texts: ... process and save ...

# --- Final Summary ---
print(f"\nFinished processing up to {MAX_ROWS} rows.")
print(f"Total texts embedded and saved to {OUTPUT_FILE}: {total_processed_count}")
if os.path.exists(OUTPUT_FILE):
    print(f"Output file size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
else:
    print("Output file was not created (possibly due to errors or no data processed).")
