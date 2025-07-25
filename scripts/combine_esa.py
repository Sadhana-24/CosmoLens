import requests
import json
import os
from datasets import load_dataset
import time
import traceback
from PIL import Image
# Import DecompressionBombError, although the check will be disabled
from PIL.Image import DecompressionBombError

# --- Configuration ---
DATASET_NAME = "Supermaxman/esa-hubble"
DATASET_SPLIT = "train"      # Adjust if the split has a different name
# No single TEXT_COLUMN, we generate text dynamically
START_INDEX = 0              # Start from the beginning
MAX_ROWS = 2706              # Process all rows (or set a smaller number for testing)
BATCH_SIZE = 50              # Adjust based on API limits and desired frequency of saves
# Dynamic filename reflecting the range and dataset
OUTPUT_FILE = f"hubble_embeddings_rows_{START_INDEX}_to_{START_INDEX + MAX_ROWS - 1}_new.jsonl"
JINA_API_URL = 'https://api.jina.ai/v1/embeddings'
JINA_MODEL = "jina-embeddings-v3" # Use a Jina v3 compatible model

# --- API Keys ---
# Add your Jina API key(s) here. The script will prompt if this list is empty.
JINA_API_KEYS = [
    "jina_b4c3dc3ab639406ca5721acc23f3215dBJSgIOxln5-jtUgXC_5_hmri1e68"
]
current_key_index = 0

# --- Pillow Decompression Bomb Limit ---
# DISABLE the check entirely as Hubble dataset has large images.
# This prevents errors during dataset iteration but removes a safety check.
Image.MAX_IMAGE_PIXELS = None

# --- Helper Function to Combine Hubble Text ---
def get_combined_text_hubble(row):
    """
    Combines relevant textual fields from an ESA Hubble dataset row
    into a single string suitable for embedding.
    """
    if not isinstance(row, dict):
        return "" # Return empty string if row is not a dictionary

    # Define the textual fields to combine for embedding context
    fields_to_combine = [
        "text",         # Primary combined title + description
        "credits",      # Attribution info
        "Type",         # Image type/subject keywords
        "Name",         # Specific names of celestial objects
        "Distance",     # Contextual info
        "Constellation",# Contextual info
        "Category"      # Broader category
    ]

    text_parts = []
    for field in fields_to_combine:
        value = row.get(field) # Use .get() for safety

        # Process only non-empty strings
        if isinstance(value, str) and value.strip():
            # Basic cleaning (e.g., for HTML entities if present)
            cleaned_value = value.strip().replace("°", " degrees ") # Example cleaning
            # Add more cleaning rules here if needed
            text_parts.append(cleaned_value)

    # Join the collected text parts with a space
    combined_text = " ".join(filter(None, text_parts))
    return combined_text

# --- Helper Function for API Call with Key Rotation ---
def get_jina_embeddings(texts_batch):
    """Sends a batch of texts to Jina API, handles key rotation and errors.
    Returns embeddings list on success (with Nones for invalid inputs),
    None if all keys fail entirely for this batch."""
    global current_key_index
    global JINA_API_KEYS

    if not JINA_API_KEYS:
        print("Error: No JINA API keys available.")
        return None # Cannot proceed without keys

    max_retries = len(JINA_API_KEYS)
    last_error = None

    # Filter out empty/invalid texts *before* sending, keeping track of original indices
    valid_texts_with_indices = [
        (idx, text) for idx, text in enumerate(texts_batch) if isinstance(text, str) and text.strip()
    ]
    # If the batch becomes empty after filtering, return Nones for all original items
    if not valid_texts_with_indices:
         # print("Warning: Skipping API request for batch with no valid text after filtering.") # Less verbose
         return [None] * len(texts_batch)

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
            "input": valid_texts_batch # Send only the valid texts
        }

        try:
            response = requests.post(JINA_API_URL, headers=headers, data=json.dumps(data), timeout=90) # Increased timeout

            # Handle specific HTTP errors indicative of key issues
            if response.status_code == 401: # Unauthorized
                error_detail = "Authentication failed (HTTP 401). Key might be invalid."
                print(f"Error with key index {key_to_try_index}: {error_detail}")
                last_error = error_detail
                continue # Try next key
            if response.status_code == 402 or response.status_code == 429: # Quota or Rate Limit
                 error_detail = "Unknown reason"
                 try: error_detail = response.json().get("detail", "No detail provided")
                 except json.JSONDecodeError: error_detail = response.text # Use raw text if JSON fails
                 print(f"Quota/Rate Limit likely issue for key index {key_to_try_index} (HTTP {response.status_code}): {error_detail}. Trying next key.")
                 last_error = f"HTTP {response.status_code}: {error_detail}"
                 if response.status_code == 429: time.sleep(5) # Wait on rate limit
                 continue # Try next key

            response.raise_for_status() # Raise HTTPError for other bad responses (4xx, 5xx)

            response_data = response.json()

            # --- Jina v3 Response Structure Check ---
            embeddings_data = []
            if isinstance(response_data, list): # Direct list response
                embeddings_data = response_data
            elif isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list): # Response wrapped in 'data'
                 embeddings_data = response_data['data']
            else:
                 error_detail = f"Unexpected response format: {response_data}"
                 print(f"Error with key {key_to_try_index}: {error_detail}")
                 last_error = error_detail
                 continue # Try next key

            # Extract embeddings, checking structure of individual items
            extracted_embeddings = []
            malformed_items = 0
            for item in embeddings_data:
                 # Check if item is a dict and has the 'embedding' key
                 if isinstance(item, dict) and 'embedding' in item and isinstance(item['embedding'], list):
                     extracted_embeddings.append(item['embedding'])
                 else:
                     malformed_items += 1
                     print(f"Warning: Malformed item in response from key {key_to_try_index}: {item}")

            if malformed_items > 0:
                print(f"Warning: Received {malformed_items} malformed embedding items using key {key_to_try_index}.")
                # Decide if this is fatal for the key - let's assume it might be transient and try next

            # Check if the number of *successfully extracted* embeddings matches the number of *valid texts sent*
            if len(extracted_embeddings) == len(valid_texts_batch):
                # print(f"Successfully obtained {len(extracted_embeddings)} embeddings using key index {key_to_try_index}.")
                current_key_index = key_to_try_index # Remember the working key index for next time
                # Reconstruct the full list matching the original batch, inserting Nones for skipped items
                full_embeddings_list = [None] * len(texts_batch)
                valid_embedding_iter = iter(extracted_embeddings)
                for original_idx, _ in valid_texts_with_indices:
                    full_embeddings_list[original_idx] = next(valid_embedding_iter)
                return full_embeddings_list
            else:
                # This case handles mismatches, potentially due to partial API failures or the malformed items check
                error_detail = f"Mismatch: Sent {len(valid_texts_batch)} valid texts, received {len(extracted_embeddings)} valid embeddings."
                print(f"Warning with key {key_to_try_index}: {error_detail}")
                last_error = error_detail
                # Treat mismatch as a potential issue with the key/API, try next key
                continue

        except requests.exceptions.Timeout:
             error_detail = "Request timed out."
             print(f"Error with key index {key_to_try_index}: {error_detail}")
             last_error = error_detail
             continue # Try next key
        except requests.exceptions.RequestException as e:
            error_detail = f"Request failed: {e}"
            print(f"Error with key index {key_to_try_index}: {error_detail}")
            last_error = error_detail
            time.sleep(1) # Small delay before trying next key
            continue # Try next key

    # If the loop finishes without returning, all keys failed for this batch
    print(f"All available API keys failed for this batch. Last error: {last_error}")
    return None # Indicate complete failure for the batch

# --- Function to Process and Save Batch ---
def process_and_save_batch(texts_batch, metadata_batch, output_file):
    """Gets embeddings for a batch, handles key prompts, and saves to file.
       Returns the number of embeddings successfully processed and saved."""
    global current_key_index
    global JINA_API_KEYS
    global should_stop_processing # Allow this function to signal a stop

    if not texts_batch:
        return 0 # Nothing to process

    embeddings = None # Ensure embeddings is reset before the attempt loop
    while embeddings is None: # Loop until embeddings are obtained or user quits
        # get_jina_embeddings returns a list (potentially with Nones) or None on total failure
        embeddings = get_jina_embeddings(texts_batch)

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
                     current_key_index = len(JINA_API_KEYS) - 1 # Try new key first next time
                else:
                    print("⚠️ Key already exists in the list.")
                # Loop will continue and retry get_jina_embeddings with the updated key list
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
            # Iterate through the embeddings list (which matches the metadata_batch length)
            for i, embedding in enumerate(embeddings):
                if embedding is not None: # Save only if an embedding was successfully generated
                    # Structure for each line in the JSONL file
                    record = {
                        "embedding": embedding,
                        "metadata": metadata_batch[i] # Match metadata by original index in the batch
                    }
                    f.write(json.dumps(record) + '\n')
                    saved_count += 1
                # else: # Optional: Log if an item was skipped due to missing embedding
                #    print(f"Skipping save for item index {i} in batch (original row: {metadata_batch[i].get('original_row_index', 'N/A')}) due to missing embedding.")

        # Print summary for the batch saving operation
        if saved_count > 0:
             print(f"   Successfully saved {saved_count} embeddings from this batch.")
        elif len(texts_batch) > 0: # We had items in the batch, but none were saved
             print(f"   No embeddings were successfully generated or saved for this batch of {len(texts_batch)} items.")
        # If texts_batch was empty, no message needed here

    except IOError as e:
        print(f"❌ Error writing batch to file {output_file}: {e}")
        print("🛑 Stopping processing due to file write error.")
        should_stop_processing = True # Signal stop
        return 0 # Indicate failure

    return saved_count # Return number successfully saved in this batch

# --- Main Processing Logic ---

# Prompt for initial API key if list is empty at the start
if not JINA_API_KEYS:
    print("\nNo Jina API keys found.")
    while not JINA_API_KEYS:
        new_key = input("Please enter at least one Jina API key to start: ").strip()
        if new_key:
            JINA_API_KEYS.append(new_key)
            current_key_index = 0
        else:
            print("No API key provided. Exiting.")
            exit()

# Print configuration summary
print("-" * 50)
print(f"Starting Embedding Process")
print(f"Dataset:          {DATASET_NAME} (Split: {DATASET_SPLIT})")
print(f"Processing Range: Start Index {START_INDEX}, Max Rows {MAX_ROWS}")
print(f"Pillow Max Pixels: {'DISABLED' if Image.MAX_IMAGE_PIXELS is None else Image.MAX_IMAGE_PIXELS}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"Output File:      {OUTPUT_FILE}")
print(f"Using API Key(s): {len(JINA_API_KEYS)} key(s) initially.")
print("-" * 50)

# --- Load Dataset ---
try:
    print("Loading dataset in streaming mode...")
    # Use trust_remote_code=True if required by the dataset's loading script
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True, trust_remote_code=True)
    print("Dataset loaded.")
except Exception as e:
    print(f"❌ Error loading dataset '{DATASET_NAME}': {e}")
    exit()

# --- Prepare Output File ---
# Remove the output file if it exists for this specific range to start fresh
if os.path.exists(OUTPUT_FILE):
    print(f"⚠️ Warning: Output file '{OUTPUT_FILE}' already exists. Removing it.")
    try:
        os.remove(OUTPUT_FILE)
    except OSError as e:
        print(f"❌ Error removing existing output file: {e}. Please remove it manually or change the filename.")
        exit()

# --- Initialize Counters and Flags ---
total_processed_count_in_run = 0 # Total embeddings saved in this execution
current_batch_texts = []         # Holds texts for the current batch
current_batch_metadata = []      # Holds metadata for the current batch
should_stop_processing = False   # Flag to signal premature stop (e.g., user quit, fatal error)
rows_processed_in_this_run = 0   # Counter for rows processed *within the specified range*
target_end_index_exclusive = START_INDEX + MAX_ROWS if MAX_ROWS is not None else float('inf') # Calculate end index

print(f"Processing rows from index {START_INDEX} up to (but not including) {target_end_index_exclusive}...")
if START_INDEX > 0:
    print(f"(Skipping rows before index {START_INDEX}...)")

row_idx = -1 # Initialize for robust error reporting in case of early failure
# --- Main Processing Loop ---
try:
    last_print_time = time.time()
    for row_idx, row_data in enumerate(dataset):

        # --- Check for stop signal FIRST ---
        if should_stop_processing:
            print("🛑 Stop signal received. Breaking main loop.")
            break

        # --- Skipping Phase ---
        if row_idx < START_INDEX:
            # Optional: Print progress less frequently during skipping phase
            if row_idx % 5000 == 0 and row_idx > 0 and time.time() - last_print_time > 10:
                print(f"   ... still skipping, currently at row {row_idx}")
                last_print_time = time.time()
            continue # Go to the next row

        # --- Termination Condition ---
        # Stop if we have processed MAX_ROWS or reached the calculated end index
        if row_idx >= target_end_index_exclusive:
             print(f"\n🏁 Reached target end index ({target_end_index_exclusive}). Processed {rows_processed_in_this_run} rows in the specified range.")
             break # Exit the main processing loop

        # --- Row Processing Start (within the desired range) ---
        # Print a message only for the very first row processed in the range
        if rows_processed_in_this_run == 0: # Note: START_INDEX <= row_idx is implicitly true here
             print(f"🚀 Starting processing data from row index {row_idx}")
        rows_processed_in_this_run += 1 # Increment count for rows processed in the target range

        # --- Generate Combined Text ---
        combined_text = get_combined_text_hubble(row_data)

        # --- Skip Row if No Text Generated ---
        if not combined_text:
            # print(f"Skipping row {row_idx} - no combined text generated.") # Optional: Log skips
            continue # Go to the next row

        # --- Prepare Metadata for this Row ---
        # Select relevant fields from the Hubble dataset row to store
        metadata = {
            "id": row_data.get("id", f"missing_id_{row_idx}"), # Primary ID
            "title": row_data.get("title", ""),             # Original title
            "credits": row_data.get("credits", ""),         # Credits field
            "name": row_data.get("Name", ""),               # Celestial object Name
            "constellation": row_data.get("Constellation", ""),# Constellation
            "category": row_data.get("Category", ""),       # Category
            "url": row_data.get("url", ""),                 # Source URL
            # Include a preview of the text actually used for embedding
            "caption": combined_text,
            "original_row_index": row_idx                   # Keep track of the original row index
        }

        # --- Add data to current batch ---
        current_batch_texts.append(combined_text)
        current_batch_metadata.append(metadata)

        # --- Process Batch when Full ---
        if len(current_batch_texts) >= BATCH_SIZE:
            print(f"\nProcessing batch of {len(current_batch_texts)} (Rows ~{row_idx - len(current_batch_texts) + 1} to {row_idx}). Total saved: {total_processed_count_in_run}...")
            # Call the function to get embeddings and save the batch
            processed_count = process_and_save_batch(current_batch_texts, current_batch_metadata, OUTPUT_FILE)
            total_processed_count_in_run += processed_count # Update total saved count

            # Clear the batch lists for the next one
            current_batch_texts = []
            current_batch_metadata = []

            # Check if processing should stop after handling the batch (e.g., due to user input or write error)
            if should_stop_processing:
                print("🛑 Stop signal received after batch processing. Breaking loop.")
                break

# --- End of Main For Loop ---

# --- Handle Loop Exit Reasons and Errors ---
except StopIteration:
     print("\n🏁 Finished iterating through the dataset stream (StopIteration).")
except KeyboardInterrupt:
     print("\n🛑 Processing interrupted by user (Ctrl+C).")
     should_stop_processing = True # Signal stop for final batch handling
except Exception as e:
     # Catch any other unexpected error during the loop or dataset iteration
     print(f"\n❌ An unexpected error occurred during dataset processing loop (around index ~{row_idx}):")
     print(f"   Error Type: {type(e).__name__}")
     print(f"   Error Details: {e}")
     print("   Traceback:")
     traceback.print_exc()
     should_stop_processing = True # Signal stop

# --- Process Final (Potentially Incomplete) Batch ---
# Process any remaining items gathered in the batch lists, unless stopped
if current_batch_texts and not should_stop_processing:
    print(f"\nProcessing final batch of {len(current_batch_texts)} items...")
    processed_count = process_and_save_batch(current_batch_texts, current_batch_metadata, OUTPUT_FILE)
    total_processed_count_in_run += processed_count
elif current_batch_texts and should_stop_processing:
    # If stopped, inform user the last batch wasn't processed
    print(f"\n⚠️ Skipping final batch of {len(current_batch_texts)} items due to stop signal.")

# --- Final Summary ---
print(f"\n--- Processing Run Summary ({DATASET_NAME}) ---")
print(f"Requested Range: Start Index {START_INDEX}, Max Rows {MAX_ROWS}")
print(f"Rows processed within range: {rows_processed_in_this_run}") # How many rows were attempted in the range
print(f"Total texts embedded and saved to '{OUTPUT_FILE}' in this run: {total_processed_count_in_run}") # How many embeddings were successfully saved

# Check output file status
if os.path.exists(OUTPUT_FILE):
    try:
        file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024*1024)
        print(f"Output file size: {file_size_mb:.2f} MB")
    except OSError as e:
        print(f"Could not get size of output file {OUTPUT_FILE}: {e}")
else:
    # File doesn't exist, check if any embeddings were expected
    if total_processed_count_in_run == 0 and rows_processed_in_this_run > 0:
         print(f"Output file '{OUTPUT_FILE}' was not created (no embeddings successfully saved).")
    elif total_processed_count_in_run == 0 and rows_processed_in_this_run == 0:
        print(f"Output file '{OUTPUT_FILE}' was not created (no rows processed in range).")
    else: # This case should ideally not happen if saving worked.
        print(f"Warning: Output file '{OUTPUT_FILE}' not found despite saving {total_processed_count_in_run} embeddings.")

print(f"Final number of API keys in list: {len(JINA_API_KEYS)}")
print("-" * 50)

# Suggest next start index if stopped prematurely
if should_stop_processing and total_processed_count_in_run > 0:
     # Suggest next start based on the *last index successfully processed* by the loop
     next_start_index = row_idx + 1 if row_idx != -1 else START_INDEX # Use last known index + 1
     print(f"\n💡 Processing stopped around index {row_idx}. To resume, you might set START_INDEX = {next_start_index} for the next run.")
elif should_stop_processing:
     print(f"\n💡 Processing stopped before any embeddings were saved in this run.")

print("Script finished.")