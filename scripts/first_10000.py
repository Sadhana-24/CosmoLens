import requests
import json
import os
from datasets import load_dataset
import time
import traceback # For better error reporting

# --- Configuration ---
DATASET_NAME = "UniverseTBD/AstroLLaVA_convos"
DATASET_SPLIT = "train"
TEXT_COLUMN = "conversation"
START_INDEX = 8501  # <<< New: Start processing from this row index (0-based)
MAX_ROWS = 1500   # <<< Process at most this many rows *starting from* START_INDEX
BATCH_SIZE = 50
# Dynamic filename reflecting the range
OUTPUT_FILE = f"astronomy_embeddings_rows_{START_INDEX}_to_{START_INDEX + MAX_ROWS - 1}.jsonl"
JINA_API_URL = 'https://api.jina.ai/v1/embeddings'
JINA_MODEL = "jina-embeddings-v3" # Correct model name for v3

# --- API Keys ---
# Start with your initial keys. More can be added interactively.
JINA_API_KEYS = [
    'jina_77ed7772b0284b51b1eafcc387f195e0DGZV3adwo0DtFHgKcWBGQNZDB_dh'
]
current_key_index = 0
if not JINA_API_KEYS:
    print("Warning: JINA_API_KEYS list is empty. You will be prompted to enter a key.")

# --- Helper Function to Convert Conversation Dict to String ---
def conversation_to_string(conversation_dict):
    """Convert the conversation dictionary to a single string for embedding"""
    if not isinstance(conversation_dict, dict):
        # print(f"Warning: Expected dict for conversation, got {type(conversation_dict)}. Returning empty string.")
        return "" # Keep output cleaner

    conversation_text = ""

    # Check preferred structure: 'conversation' key with list of dicts
    if 'conversation' in conversation_dict and isinstance(conversation_dict['conversation'], list):
        texts = []
        for msg_data in conversation_dict['conversation']:
            if isinstance(msg_data, dict) and 'value' in msg_data and isinstance(msg_data['value'], str):
                texts.append(msg_data['value'].strip())
            elif isinstance(msg_data, str):
                 texts.append(msg_data.strip())
        conversation_text = " ".join(filter(None, texts))

    # Fallback: Check structure with 'from' and 'value' lists
    elif all(k in conversation_dict and isinstance(conversation_dict[k], list) for k in ["from", "value"]):
        texts = []
        for msg in conversation_dict.get("value", []):
            if isinstance(msg, str):
                texts.append(msg.strip())
        conversation_text = " ".join(filter(None, texts))

    else:
        # print(f"Warning: Conversation dict has unknown structure: {list(conversation_dict.keys())}. Returning empty string.")
        return "" # Keep output cleaner

    return conversation_text

# --- Helper Function for API Call with Key Rotation ---
def get_jina_embeddings(texts_batch):
    """Sends a batch of texts to Jina API, handles key rotation and errors.
    Returns embeddings list on success, None if all keys fail for this batch."""
    global current_key_index
    global JINA_API_KEYS

    if not JINA_API_KEYS:
        print("Error: No JINA API keys available.")
        return None

    max_retries = len(JINA_API_KEYS)
    last_error = None

    # Filter out empty/invalid texts *before* sending
    valid_texts_with_indices = [
        (idx, text) for idx, text in enumerate(texts_batch) if isinstance(text, str) and text.strip()
    ]
    if not valid_texts_with_indices:
         print("Warning: Skipping batch because it contains no valid text after filtering.")
         return [] # Return empty list as no embeddings needed/generated

    valid_texts_batch = [text for idx, text in valid_texts_with_indices]


    for attempt in range(max_retries):
        key_to_try_index = (current_key_index + attempt) % len(JINA_API_KEYS)
        api_key = JINA_API_KEYS[key_to_try_index]

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            "model": JINA_MODEL,
            "input": valid_texts_batch
        }

        # print(f"Attempting API call with key index {key_to_try_index}...") # Can be noisy
        try:
            response = requests.post(JINA_API_URL, headers=headers, data=json.dumps(data), timeout=90)

            if response.status_code == 401:
                error_detail = f"Authentication failed (HTTP 401). Key might be invalid."
                print(f"Error with key index {key_to_try_index}: {error_detail}")
                last_error = error_detail
                continue
            if response.status_code == 402 or response.status_code == 429:
                 error_detail = "Unknown"
                 try: error_detail = response.json().get("detail", "No detail provided")
                 except json.JSONDecodeError: error_detail = response.text
                 print(f"Quota/Rate Limit likely issue for key index {key_to_try_index} (HTTP {response.status_code}): {error_detail}. Trying next key.")
                 last_error = f"HTTP {response.status_code}: {error_detail}"
                 if response.status_code == 429:
                     # print("Rate limit hit. Waiting 5 seconds before trying next key...")
                     time.sleep(5) # Simple backoff
                 continue

            response.raise_for_status() # Raise HTTPError for other bad responses (4xx, 5xx)

            response_data = response.json()
            embeddings_data = []
            if isinstance(response_data, list):
                embeddings_data = response_data
            elif isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                 embeddings_data = response_data['data']
            else:
                 error_detail = f"Unexpected response format: {response_data}"
                 print(f"Error with key {key_to_try_index}: {error_detail}")
                 last_error = error_detail
                 continue

            # Extract embeddings, checking structure
            extracted_embeddings = []
            malformed_items = 0
            for item in embeddings_data:
                 if isinstance(item, dict) and 'embedding' in item:
                     extracted_embeddings.append(item['embedding'])
                 else:
                     malformed_items += 1

            if malformed_items > 0:
                print(f"Warning: Received {malformed_items} malformed items in response from key {key_to_try_index}.")
                # Decide if this is fatal for the key - let's assume it might be transient and try next

            if len(extracted_embeddings) == len(valid_texts_batch):
                # print(f"Successfully obtained {len(extracted_embeddings)} embeddings using key index {key_to_try_index}.")
                # Success! Update the index for the *next* batch attempt
                current_key_index = key_to_try_index
                # Reconstruct the full list including placeholders for skipped texts
                full_embeddings_list = [None] * len(texts_batch)
                valid_embedding_iter = iter(extracted_embeddings)
                for original_idx, _ in valid_texts_with_indices:
                    full_embeddings_list[original_idx] = next(valid_embedding_iter)
                return full_embeddings_list
            else:
                error_detail = f"Mismatch in expected ({len(valid_texts_batch)}) and received ({len(extracted_embeddings)}) embeddings."
                print(f"Warning with key {key_to_try_index}: {error_detail}")
                last_error = error_detail
                continue

        except requests.exceptions.Timeout:
             error_detail = "Request timed out."
             print(f"Error with key index {key_to_try_index}: {error_detail}")
             last_error = error_detail
             continue
        except requests.exceptions.RequestException as e:
            error_detail = f"Request failed: {e}"
            print(f"Error with key index {key_to_try_index}: {error_detail}")
            last_error = error_detail
            time.sleep(1)
            continue

    print(f"All available API keys failed for this batch. Last error: {last_error}")
    return None

# --- Function to Process and Save Batch ---
# Encapsulates the logic involving API calls and file writing for a batch
def process_and_save_batch(texts_batch, metadata_batch, output_file):
    """Gets embeddings for a batch, handles key prompts, and saves to file.
       Returns the number of embeddings successfully processed and saved."""
    global current_key_index
    global JINA_API_KEYS
    global should_stop_processing # Allow this function to signal a stop

    if not texts_batch:
        return 0 # Nothing to process

    embeddings = None
    while embeddings is None: # Loop until embeddings are obtained or user quits
        embeddings = get_jina_embeddings(texts_batch) # Returns list (potentially with Nones) or None

        if embeddings is None: # This means all current keys failed entirely for the batch
            print("\n-----------------------------------------------------")
            print("🔴 All available API keys failed for the current batch.")
            print(f"   Current keys ({len(JINA_API_KEYS)} total).")
            print("   This might be due to quota limits or invalid keys.")
            new_key = input("🔑 Please enter a new Jina API key to continue, or press Enter to stop: ").strip()
            print("-----------------------------------------------------")

            if new_key:
                if new_key not in JINA_API_KEYS:
                     JINA_API_KEYS.append(new_key)
                     print(f"✅ Added new key. Total keys: {len(JINA_API_KEYS)}")
                     # Set index to try the new key first next time
                     current_key_index = len(JINA_API_KEYS) - 1
                else:
                    print("⚠️ Key already exists in the list. Please provide a different one or press Enter to stop.")
                # Loop will continue and retry get_jina_embeddings
            else:
                print("🛑 No new key provided. Stopping processing.")
                should_stop_processing = True
                return 0 # Indicate 0 processed as we are stopping

    # --- Saving Logic ---
    saved_count = 0
    if should_stop_processing: # Check if stop was requested during key prompt
         return 0

    # Embeddings is now a list, potentially with None for failed/skipped items
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for i, embedding in enumerate(embeddings):
                if embedding is not None: # Save only successful embeddings
                    record = {
                        "embedding": embedding,
                        "metadata": metadata_batch[i] # Match metadata by original index
                    }
                    f.write(json.dumps(record) + '\n')
                    saved_count += 1
                else:
                    # Log which item failed if needed (metadata_batch[i]['id'] or ['original_row_index'])
                    # print(f"Skipping save for item index {i} in batch (original row: {metadata_batch[i]['original_row_index']}) due to missing embedding.")
                    pass # Silently skip items that didn't get an embedding

        if saved_count > 0:
             print(f"   Successfully saved {saved_count} embeddings from this batch.")
        elif len(texts_batch) > 0 : # We had texts, but none were saved
             print(f"   No embeddings were successfully generated or saved for this batch of {len(texts_batch)} items.")


    except IOError as e:
        print(f"❌ Error writing batch to file {output_file}: {e}")
        print("🛑 Stopping processing due to file write error.")
        should_stop_processing = True # Stop processing if cannot write
        return 0 # Indicate failure

    return saved_count # Return number successfully saved

# --- Main Processing Logic ---

# Prompt for initial API key if list is empty
while not JINA_API_KEYS:
    print("\nNo Jina API keys found.")
    new_key = input("Please enter a Jina API key to start: ").strip()
    if new_key:
        JINA_API_KEYS.append(new_key)
        current_key_index = 0
    else:
        print("No API key provided. Exiting.")
        exit()

print(f"Using initial API key(s): {len(JINA_API_KEYS)}")
print("-" * 50)
print(f"Dataset:          {DATASET_NAME} (Split: {DATASET_SPLIT})")
print(f"Processing Range: Start Index {START_INDEX}, Max Rows {MAX_ROWS}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"Output File:      {OUTPUT_FILE}")
print("-" * 50)

try:
    # Load dataset in streaming mode
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True, trust_remote_code=True)
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Check if output file exists and append or start fresh? Let's start fresh for a defined range.
if os.path.exists(OUTPUT_FILE):
    print(f"⚠️ Warning: Output file '{OUTPUT_FILE}' already exists. Removing it to start fresh for this range.")
    try:
        os.remove(OUTPUT_FILE)
    except OSError as e:
        print(f"❌ Error removing existing output file: {e}. Please remove it manually or change the filename.")
        exit()

total_processed_count_in_run = 0
current_batch_texts = []
current_batch_metadata = []
should_stop_processing = False # Flag to signal premature stop
rows_processed_in_this_run = 0 # Counter for rows actually processed (after START_INDEX)
target_end_index_exclusive = START_INDEX + MAX_ROWS if MAX_ROWS is not None else float('inf') # Calculate end index

print(f"Starting embedding process...")
if START_INDEX > 0:
    print(f"Skipping rows before index {START_INDEX}...")

# Process the dataset in streaming mode
try:
    last_print_time = time.time()
    for row_idx, row in enumerate(dataset):

        if should_stop_processing:
            print("🛑 Stopping processing as requested.")
            break

        # --- Skipping Phase ---
        if row_idx < START_INDEX:
            # Optional: Print progress less frequently during skipping
            if row_idx % 5000 == 0 and time.time() - last_print_time > 10:
                print(f"   ... still skipping, currently at row {row_idx}")
                last_print_time = time.time()
            continue

        # --- Termination Condition ---
        if row_idx >= target_end_index_exclusive:
             print(f"\n🏁 Reached target end index ({target_end_index_exclusive}). Processed {rows_processed_in_this_run} rows in the specified range.")
             break

        # --- Row Processing ---
        if rows_processed_in_this_run == 0 and START_INDEX > 0: # First row after skipping
             print(f"🚀 Starting processing from row index {row_idx}")

        rows_processed_in_this_run += 1 # Count this row towards the MAX_ROWS limit

        # Safely access 'conversation' column
        conversation_data = row.get(TEXT_COLUMN)
        if conversation_data is None:
            # print(f"Skipping row {row_idx} - '{TEXT_COLUMN}' column missing or None.")
            continue

        conversation_text = conversation_to_string(conversation_data)

        # Skip empty conversations
        if not conversation_text or not conversation_text.strip():
            # print(f"Skipping row {row_idx} - empty or invalid conversation content")
            continue

        # Prepare metadata
        metadata = {
            "id": row.get("id", f"missing_id_{row_idx}"),
            "caption": row.get("caption", ""),
            "url": row.get("url", ""),
            "conversation_text_preview": conversation_text[:100] + "..." if len(conversation_text) > 100 else conversation_text,
            "original_row_index": row_idx
        }

        # Add to current batch
        current_batch_texts.append(conversation_text)
        current_batch_metadata.append(metadata)

        # --- Process Batch ---
        if len(current_batch_texts) >= BATCH_SIZE:
            print(f"\nProcessing batch of {len(current_batch_texts)} (Rows ~{row_idx - len(current_batch_texts) + 1} to {row_idx}). Total processed in run: {total_processed_count_in_run}...")
            processed_count = process_and_save_batch(current_batch_texts, current_batch_metadata, OUTPUT_FILE)
            total_processed_count_in_run += processed_count
            # Clear the batch
            current_batch_texts = []
            current_batch_metadata = []
            if should_stop_processing: # Check if stop was triggered during batch processing
                break

except StopIteration:
     print("\n🏁 Finished iterating through the dataset stream.")
except KeyboardInterrupt:
     print("\n🛑 Processing interrupted by user (Ctrl+C).")
     should_stop_processing = True # Signal stop
except Exception as e:
     print(f"\n❌ An unexpected error occurred during dataset processing loop at row index ~{row_idx}:")
     print(f"   Error Type: {type(e).__name__}")
     print(f"   Error Details: {e}")
     print("   Traceback:")
     traceback.print_exc()
     should_stop_processing = True # Stop on unexpected errors

# --- Process Final Batch ---
# Need to process any remaining items after the loop finishes or breaks
if current_batch_texts and not should_stop_processing:
    print(f"\nProcessing final batch of {len(current_batch_texts)} items...")
    processed_count = process_and_save_batch(current_batch_texts, current_batch_metadata, OUTPUT_FILE)
    total_processed_count_in_run += processed_count
elif current_batch_texts and should_stop_processing:
    print(f"\n⚠️ Skipping final batch of {len(current_batch_texts)} items due to processing stop signal.")


# --- Final Summary ---
print(f"\n--- Processing Run Summary ---")
print(f"Requested Range: Start Index {START_INDEX}, Max Rows {MAX_ROWS}")
actual_rows_iterated = rows_processed_in_this_run # How many rows were actually encountered in the range
print(f"Rows iterated within range: {actual_rows_iterated}")
print(f"Total texts embedded and saved to '{OUTPUT_FILE}' in this run: {total_processed_count_in_run}")
if os.path.exists(OUTPUT_FILE):
    try:
        file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024*1024)
        print(f"Output file size: {file_size_mb:.2f} MB")
    except OSError as e:
        print(f"Could not get size of output file {OUTPUT_FILE}: {e}")
else:
    if total_processed_count_in_run == 0:
         print(f"Output file '{OUTPUT_FILE}' was not created or is empty (no embeddings saved).")
    else:
        print(f"Warning: Output file '{OUTPUT_FILE}' not found despite saving {total_processed_count_in_run} embeddings.")

print(f"Final number of API keys in list: {len(JINA_API_KEYS)}")
print("-" * 50)

if should_stop_processing and total_processed_count_in_run > 0:
    next_start_index = START_INDEX + total_processed_count_in_run # This might not be perfectly accurate if rows were skipped *within* the processed range
    # A safer next start index might be the last successfully processed row's index + 1
    # This requires tracking the last successful index. For simplicity, let's suggest based on count.
    print(f"\n💡 Processing stopped. To resume, you might want to set START_INDEX = {START_INDEX + rows_processed_in_this_run} for the next run (adjust based on logs).")