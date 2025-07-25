import os
from re import split
# import secrets
from modal import App, Image, Volume, Secret
from pathlib import Path

# import subprocess
# from sympy import sec
# Create the Modal app
app = App("sft-trainer")

# Create secrets for API keys
# These will be available as environment variables in the container
huggingface_secret = Secret.from_name("huggingface-secret")
wandb_secret = Secret.from_name("wandb-secret")

# Create volumes for persistent storage
model_volume = Volume.from_name("grpo-models-vol", create_if_missing=True)
runs_volume = Volume.from_name("grpo-runs-vol", create_if_missing=True)
data_volume = Volume.from_name("grpo-data-vol", create_if_missing=True)


# Configure volume mounting points
VOLUME_CONFIG = {
    "/pretrained": model_volume,
    "/runs": runs_volume,
    "/data": data_volume,
}

CPU_IMAGE = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "tqdm",
        "pillow",
        "transformers",
        "datasets",
        "huggingface_hub",
    )
)
# Set up base image with dependencies
GPU_IMAGE = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("cupy-cuda12x")
    .pip_install(
        "pyyaml",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "wandb",
        "huggingface_hub",
        "bitsandbytes",
        "qwen-vl-utils"
    )
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
)


# SWIFT_GPU_IMAGE = (
#     Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
#     .apt_install("git")
#     .pip_install("cupy-cuda12x")
#     .pip_install(
#         "pyyaml",
#         "transformers",
#         "datasets",
#         "accelerate",
#         "peft",
#         "trl",
#         "wandb",
#         "huggingface_hub",
#         "bitsandbytes",
#         "qwen-vl-utils"
#     )
#     .pip_install(
#         "vllm==0.7.2",
#         "huggingface_hub[hf_transfer]==0.26.2",
#         "flashinfer-python==0.2.0.post2",  # pinning, very unstable
#         extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
#     )
#     .run_commands("pip install flash-attn --no-build-isolation")
#     .run_commands("pip install 'ms-swift[all]' -U")  # Added MS Swift installation
# )

SWIFT_GPU_IMAGE = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    # Install system dependencies for OpenCV and other libraries
    .apt_install(
        "git",
        "libgl1-mesa-glx",  # Required for OpenCV (libGL.so.1)
        "libglib2.0-0",     # Common dependency
        "libsm6",           # X11 dependencies
        "libxext6",
        "libxrender-dev", 
        "libfontconfig1",   # Font configuration
        "libfreetype6"      # Font rendering
    )
    .pip_install("cupy-cuda12x")
    .pip_install(
        "pyyaml",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "wandb",
        "huggingface_hub",
        "bitsandbytes",
        "qwen-vl-utils",
        "opencv-python-headless"  # Use headless version of OpenCV
    )
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .run_commands("pip install 'ms-swift[all]' -U")  # Added MS Swift installation
)

# Time constants
HOURS = 60 * 60
MINUTES = 60

# Environment variables
GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100-80gb:2")
if len(GPU_CONFIG.split(":")) <= 1:
    N_GPUS = int(os.environ.get("N_GPUS", 1))
    GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"
SINGLE_GPU_CONFIG = os.environ.get("SINGLE_GPU_CONFIG", "L40S:1")

@app.function(
    image=CPU_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
)
def data_prep():
    import logging
    import os
    import sys
    import json
    import time
    from PIL import Image as PILImage
    from typing import Optional

    import datasets
    import transformers
    from datasets import load_dataset
    from transformers import set_seed
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Hard-coded configuration
    DATASET_NAME = "v1v1d/vivid_layout"
    IMAGE_OUTPUT_DIR = "/data/processed_images"
    PROCESSED_DATA_PATH = "/data/processed_dataset"
    EVAL_SAMPLES = 10  # Number of samples to use for evaluation
    SEED = 42
    
    # Install tqdm if not already available
    try:
        import tqdm
    except ImportError:
        logger.info("Installing tqdm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        import tqdm
        
    from tqdm.auto import tqdm

    # Set seed for reproducibility
    set_seed(SEED)

    # Create output directories
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    logger.info(f"Starting data preparation for {DATASET_NAME}")
    logger.info(f"Images will be stored in {IMAGE_OUTPUT_DIR}")
    logger.info(f"Processed dataset will be stored in {PROCESSED_DATA_PATH}")

    def process_dataset():
        """Process the dataset and save it for training."""
        # Start timer for overall processing
        start_time = time.time()
        
        logger.info(f"Loading dataset from Hugging Face: {DATASET_NAME}")
        with tqdm(total=1, desc="Loading dataset", unit="dataset") as pbar:
            # Load dataset directly from Hugging Face
            raw_dataset = load_dataset(DATASET_NAME)
            pbar.update(1)
        
        # Use only the train split
        raw_data = raw_dataset["train"]
        logger.info(f"Loaded {len(raw_data)} samples from {DATASET_NAME}")
        
        # Create shuffled indices for train/eval split
        import numpy as np
        np.random.seed(SEED)
        indices = np.random.permutation(len(raw_data))
        
        # Use the last EVAL_SAMPLES for evaluation
        eval_indices = indices[-EVAL_SAMPLES:].tolist()
        train_indices = indices[:-EVAL_SAMPLES].tolist()
        
        logger.info(f"Created split: {len(train_indices)} training samples, {len(eval_indices)} evaluation samples")
        
        # Initialize empty lists to store processed data
        processed_train_data = []
        processed_eval_data = []
        
        # Function to process a single item (will be used with ThreadPoolExecutor)
        def process_item(args):
            i, idx, is_train = args
            try:
                item = raw_data[idx]
                
                # Extract layout information from the JSON string
                layout_str = item["layout"]
                
                try:
                    layout_json = json.loads(layout_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON for item {idx}, skipping")
                    return None
                
                # Extract page image from the dataset
                img = item['image']
                
                # Generate a unique filename
                pdf_name = item.get('pdf_name', f"doc_{idx}")
                page_num = item.get('page_number', 0)
                image_filename = f"{pdf_name.replace('/', '_')}_page_{page_num}.png"
                image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                
                # Save image if it doesn't exist yet
                if not os.path.exists(image_path):
                    try:
                        img.save(image_path)
                    except Exception as e:
                        logger.warning(f"Failed to save image for item {idx}: {e}, skipping")
                        return None
                
                # Format the bounding box data as required for training using your format
                bboxes = []
                for box in layout_json:
                    try:
                        # Check if your data structure matches the expected format
                        if isinstance(box, dict) and 'coordinates' in box:
                            # Use your format with 'coordinates'
                            x1, y1, x2, y2 = box["coordinates"]
                            label = box.get("type", "")
                        elif isinstance(box, dict) and 'bbox' in box:
                            # Use the original format with 'bbox'
                            x1, y1, x2, y2 = box["bbox"]
                            label = box.get("label", "")
                        else:
                            # Skip if format doesn't match
                            continue
                            
                        bboxes.append({
                            "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                            "label": label
                        })
                    except (ValueError, TypeError, KeyError) as e:
                        # Log and continue if there's an issue with a specific box
                        logger.debug(f"Error processing box in item {idx}: {e}")
                        continue
                
                # Create formatted example
                example = {
                    "image": image_path,
                    "problem": "Identify and annotate all the elements in this document with bounding boxes.",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Extract the document layout including ['title', 'text', 'plain_text', 'figure', 'figure_caption', 'table', 'table_caption', 'table_footnote', 'isolate_formula', 'embedding', 'isolated'] For each element, provide a bounding box [x1, y1, x2, y2] coordinates and a label indicating type of content.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{image_path}"},
                                {"type": "text", "text": "Format your response as a JSON array where each item includes 'bbox_2d' and 'label' fields."},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": f'```json\n{json.dumps(bboxes, indent=2)}\n```',
                        }
                    ]
                }
                
                return (example, is_train)
            
            except Exception as e:
                logger.warning(f"Failed to process item {idx}: {e}, skipping")
                return None
        
        # Set up items to process with TrainPool
        items_to_process = [(i, idx, True) for i, idx in enumerate(train_indices)]
        items_to_process.extend([(i + len(train_indices), idx, False) for i, idx in enumerate(eval_indices)])
        
        # Use ThreadPoolExecutor to parallelize processing
        from concurrent.futures import ThreadPoolExecutor
        
        # Determine the number of threads based on CPU count and IO-bound nature of the task
        num_workers = min(32, os.cpu_count() * 4)  # Use more threads since the task is I/O bound
        logger.info(f"Processing data with {num_workers} workers")
        
        # Process items with progress bar
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Map each item to an executor and track with tqdm
            results = list(tqdm(
                executor.map(process_item, items_to_process), 
                total=len(items_to_process),
                desc="Processing dataset",
                unit="samples"
            ))
        
        # Filter out None results (failed items) and sort into train/eval
        for result in results:
            if result is not None:
                example, is_train = result
                if is_train:
                    processed_train_data.append(example)
                else:
                    processed_eval_data.append(example)
        
        logger.info(f"Successfully processed {len(processed_train_data)} training and {len(processed_eval_data)} evaluation samples")
        logger.info(f"Skipped {len(items_to_process) - len(processed_train_data) - len(processed_eval_data)} samples due to errors")
        
        # Save processed training data
        train_data_file = os.path.join(PROCESSED_DATA_PATH, "train_data.json")
        logger.info(f"Saving {len(processed_train_data)} training examples to {train_data_file}...")
        
        with tqdm(total=1, desc="Saving training data", unit="file") as pbar:
            with open(train_data_file, "w") as f:
                json.dump(processed_train_data, f)
            pbar.update(1)
            
        logger.info(f"Successfully saved {len(processed_train_data)} training examples")
        
        # Save processed evaluation data
        eval_data_file = os.path.join(PROCESSED_DATA_PATH, "eval_data.json")
        logger.info(f"Saving {len(processed_eval_data)} evaluation examples to {eval_data_file}...")
        
        with tqdm(total=1, desc="Saving evaluation data", unit="file") as pbar:
            with open(eval_data_file, "w") as f:
                json.dump(processed_eval_data, f)
            pbar.update(1)
            
        logger.info(f"Successfully saved {len(processed_eval_data)} evaluation examples")
        
        # Save metadata about the dataset
        metadata = {
            "dataset_name": DATASET_NAME,
            "num_train_samples": len(processed_train_data),
            "num_eval_samples": len(processed_eval_data),
            "image_dir": IMAGE_OUTPUT_DIR,
            "train_data_file": train_data_file,
            "eval_data_file": eval_data_file,
            "seed": SEED,
        }
        
        metadata_file = os.path.join(PROCESSED_DATA_PATH, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
            
        logger.info(f"Saved dataset metadata to {metadata_file}")
        
        # Commit volumes to persist changes with progress tracking
        logger.info("Committing changes to data volume...")
        with tqdm(total=1, desc="Committing to volume", unit="commit") as pbar:
            data_volume.commit()
            pbar.update(1)
        
        # Calculate and log processing time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Processing speed: {(len(processed_train_data) + len(processed_eval_data))/total_time:.2f} samples/second")
        
        return len(processed_train_data), len(processed_eval_data)
    
    try:
        num_train, num_eval = process_dataset()
        
        logger.info("Data preparation completed successfully.")
        logger.info(f"Train samples: {num_train}")
        logger.info(f"Evaluation samples: {num_eval}")
        
        return {"train_samples": num_train, "eval_samples": num_eval}
    except Exception as e:
        logger.error(f"Data preparation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


@app.function(
    image=CPU_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
)
def data_prep_swift():
    import logging
    import os
    import sys
    import json
    import time
    from datasets import load_dataset, Dataset
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define paths
    DATASET_NAME = "v1v1d/vivid_layout"
    OUTPUT_DIR = "/data/swift_output"
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    JSONL_DIR = os.path.join(OUTPUT_DIR, "jsonl")
    
    # Create output directories
    dirs_to_create = [OUTPUT_DIR, IMAGE_DIR, JSONL_DIR]
    
    with tqdm(total=len(dirs_to_create), desc="Creating directories") as pbar:
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            pbar.update(1)
    
    logger.info(f"Loading dataset {DATASET_NAME}")
    images_dataset = load_dataset(DATASET_NAME)
    train_dataset = images_dataset["train"]
    logger.info(f"Loaded {len(train_dataset)} examples")

    def process_single_item(item, idx, pbar=None):
        """Process a single dataset item"""
        try:
            # Save the image
            img = item['image']
            pdf_name = item.get('pdf_name', f"doc_{idx}")
            page_num = item.get('page_number', 0)
            image_filename = f"{pdf_name.replace('/', '_')}_page_{page_num}.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            
            # Save image if it doesn't exist yet
            if not os.path.exists(image_path):
                img.save(image_path)
            
            # Convert image path to Unix style for consistency
            unix_style_path = image_path.replace(os.sep, '/')
            
            # Extract layout JSON
            layout_str = item["layout"]
            try:
                layout_json = json.loads(layout_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON for item {idx}, skipping")
                return None

            # Format bounding boxes for content
            bboxes_content = []
            for box in layout_json:
                try:
                    if isinstance(box, dict) and 'coordinates' in box:
                        x1, y1, x2, y2 = box["coordinates"]
                        label = box.get("type", "unknown")
                    elif isinstance(box, dict) and 'bbox' in box:
                        x1, y1, x2, y2 = box["bbox"]
                        label = box.get("label", "unknown")
                    else:
                        continue
                        
                    bboxes_content.append({
                        "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                        "label": label
                    })
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug(f"Error processing box in item {idx}: {e}")
                    continue
            
            # Create the Swift-compatible format
            transformed_item = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "Extract the document layout including ['title', 'text', 'plain_text', 'figure', 'figure_caption', 'table', 'table_caption', 'table_footnote', 'isolate_formula', 'embedding', 'isolated'] For each element, provide a bounding box [x1, y1, x2, y2] coordinates and a label indicating type of content."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<image>Format your response as a JSON array where each item includes 'bbox_2d' and 'label' fields."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": f"```json\n{json.dumps(bboxes_content, indent=2)}\n```"
                    }
                ],
                "images": [unix_style_path]
            }
            
            # Save individual JSONL file
            jsonl_path = os.path.join(JSONL_DIR, f"{pdf_name.replace('/', '_')}_page_{page_num}.jsonl")
            with open(jsonl_path, 'w') as f:
                f.write(json.dumps(transformed_item))
            
            result = {
                "item": transformed_item,
                "pdf_name": pdf_name,
                "page_num": page_num
            }

        finally:
            # Update progress bar if provided
            if pbar is not None:
                pbar.update(1)
                
        return result

    # Process the dataset
    logger.info(f"Processing {len(train_dataset)} items")
    transformed_data = []
    
    # Determine number of workers
    import multiprocessing
    num_workers = min(multiprocessing.cpu_count(), len(train_dataset))
    
    # Use ThreadPoolExecutor
    with tqdm(total=len(train_dataset), desc="Processing with ThreadPoolExecutor") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks to the executor
            futures = {
                executor.submit(
                    process_single_item, 
                    train_dataset[i], 
                    i, 
                    pbar
                ): i for i in range(len(train_dataset))
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        transformed_data.append(result["item"])
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Error processing item {idx}: {e}")
    
    # Save all data as single JSONL file
    combined_jsonl_path = os.path.join(OUTPUT_DIR, "transformed_data.jsonl")
    logger.info(f"Saving combined data to {combined_jsonl_path}")
    
    with tqdm(total=len(transformed_data), desc="Saving combined JSONL") as pbar:
        with open(combined_jsonl_path, 'w') as f:
            for item in transformed_data:
                f.write(json.dumps(item) + '\n')
                pbar.update(1)
    
    # Create dataset info
    dataset_info = {
        "dataset_name": DATASET_NAME,
        "num_samples": len(transformed_data),
        "output_dir": OUTPUT_DIR,
        "combined_jsonl_path": combined_jsonl_path
    }
    
    # Save dataset info
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Successfully processed {len(transformed_data)} examples")
    logger.info(f"Dataset prepared for Swift at {OUTPUT_DIR}")
    
    # Make sure changes are persisted
    data_volume.commit()
    
    return {"samples_processed": len(transformed_data), "output_dir": OUTPUT_DIR}

# @app.function(
#     image=GPU_IMAGE,
#     gpu=SINGLE_GPU_CONFIG,
#     volumes=VOLUME_CONFIG,
#     timeout=24 * HOURS,
#     secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
# )
# def train():
#     import logging
#     import os
#     import sys
#     import json
#     from PIL import Image
#     from typing import Optional

#     import datasets
#     import torch
#     from torch.utils.data import Dataset, DataLoader
#     import transformers
#     from transformers import AutoProcessor, set_seed, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
#     from transformers.trainer_utils import get_last_checkpoint
#     from transformers import Trainer, TrainingArguments
#     from peft import LoraConfig
#     from trl import SFTTrainer
#     os.environ["WANDB_PROJECT"] = "Qwen2.5VL-finetune"
    
#     # Configure logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     datasets.utils.logging.set_verbosity(logging.INFO)
#     transformers.utils.logging.set_verbosity(logging.INFO)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     # Hard-coded configuration
#     MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
#     PROCESSED_DATA_PATH = "/data/processed_dataset"
#     OUTPUT_DIR = "/runs/Qwen2.5-VL-vivid-layout"
#     LEARNING_RATE = 2e-5
#     NUM_TRAIN_EPOCHS = 1
#     MAX_SEQ_LENGTH = 4096
#     PER_DEVICE_TRAIN_BATCH_SIZE = 1
#     GRADIENT_ACCUMULATION_STEPS = 1
#     USE_GRADIENT_CHECKPOINTING = True
#     USE_BF16 = True
#     LOGGING_STEPS = 5
#     EVAL_STRATEGY = "steps"
#     EVAL_STEPS = 100
#     SEED = 42

#     # Set seed for reproducibility
#     set_seed(SEED)

#     # Create output directories
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Define the process_vision_info function that was imported from qwen_vl_utils
#     def process_vision_info(messages):
#         """Process vision information from messages."""
#         images = []
#         videos = []
        
#         for message in messages:
#             if not isinstance(message.get("content", ""), list):
#                 continue
            
#             for content in message["content"]:
#                 if not isinstance(content, dict):
#                     continue
                    
#                 if content.get("type") == "image" and "image" in content:
#                     image_path = content["image"]
#                     if image_path.startswith("file://"):
#                         image_path = image_path[7:]
                    
#                     try:
#                         img = Image.open(image_path)
#                         images.append(img)
#                     except Exception as e:
#                         logger.error(f"Error loading image {image_path}: {e}")
                
#                 elif content.get("type") == "video" and "video" in content:
#                     # Video processing if needed
#                     videos.append(content["video"])
        
#         return images, videos

#     class VividLayoutDataset(Dataset):
#         def __init__(self, processed_data_path, split="train"):
#             super(VividLayoutDataset, self).__init__()
            
#             # Load the processed data
#             if split == "train":
#                 processed_data_file = os.path.join(processed_data_path, "train_data.json")
#             else:
#                 processed_data_file = os.path.join(processed_data_path, "eval_data.json")
            
#             if not os.path.exists(processed_data_file):
#                 raise FileNotFoundError(f"Processed data file not found at {processed_data_file}. Run data_prep first.")
            
#             with open(processed_data_file, "r") as f:
#                 self.processed_data = json.load(f)
                    
#             logger.info(f"Loaded {len(self.processed_data)} processed samples for {split} from {processed_data_file}")
            
#             # Load metadata
#             metadata_file = os.path.join(processed_data_path, "metadata.json")
#             if os.path.exists(metadata_file):
#                 with open(metadata_file, "r") as f:
#                     self.metadata = json.load(f)
#                 logger.info(f"Dataset metadata: {self.metadata}")
        
#         def __len__(self):
#             return len(self.processed_data)

#         def __getitem__(self, i):
#             return self.processed_data[i]


#     def training():
#         logger.info("*** Initializing processor ***")
#         processor = AutoProcessor.from_pretrained(
#             MODEL_NAME, trust_remote_code=True
#         )
#         logger.info("Using AutoProcessor for vision-language model.")
        
#         if hasattr(processor, "pad_token") and processor.pad_token is None:
#             processor.pad_token = processor.eos_token
#         elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
#             processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
#         logger.info("*** Loading processed dataset ***")
#         train_dataset = VividLayoutDataset(PROCESSED_DATA_PATH, split="train")
#         eval_dataset = VividLayoutDataset(PROCESSED_DATA_PATH, split="eval")
        
#         # Check for last checkpoint
#         last_checkpoint = None
#         if os.path.isdir(OUTPUT_DIR):
#             last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
#         if last_checkpoint is not None:
#             logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
        
#         logger.info("*** Initializing model ***")
#         torch_dtype = torch.bfloat16 if USE_BF16 else torch.float32
        
#         model_kwargs = dict(
#             trust_remote_code=True,
#             torch_dtype=torch_dtype,
#             use_cache=False if USE_GRADIENT_CHECKPOINTING else True,
#         )
        
#         # Load the model
#         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             MODEL_NAME, **model_kwargs
#         )
        
#         # Define data collator function
#         def collate_fn(examples):
#             texts = [
#                 processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
#                 for example in examples
#             ]
#             image_inputs = []
#             for example in examples:
#                 imgs, vids = process_vision_info(example["messages"])
#                 image_inputs.append(imgs)
#             batch = processor(
#                 text=texts,
#                 images=image_inputs,
#                 return_tensors="pt",
#                 padding=True,
#             )
#             labels = batch["input_ids"].clone()
#             labels[labels == processor.tokenizer.pad_token_id] = -100
            
#             # Handle image token ID if available
#             if hasattr(processor, "image_token"):
#                 image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
#                 labels[labels == image_token_id] = -100
            
#             batch["labels"] = labels
#             return batch
        
#         peft_config = LoraConfig(
#             r=32,
#             lora_alpha=64,
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
#             task_type="CAUSAL_LM",
#             lora_dropout=0.05,
#         )
        
#         # Configure training arguments
#         training_args = TrainingArguments(
#             output_dir=OUTPUT_DIR,
#             learning_rate=LEARNING_RATE,
#             num_train_epochs=NUM_TRAIN_EPOCHS,
#             per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#             gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#             gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
#             bf16=USE_BF16,
#             logging_steps=LOGGING_STEPS,
#             evaluation_strategy=EVAL_STRATEGY,
#             eval_steps=EVAL_STEPS,
#             save_strategy="steps",
#             save_steps=EVAL_STEPS,
#             save_total_limit=3,
#             load_best_model_at_end=True,
#             # Not using the default column remover since our dataset doesn't support map
#             remove_unused_columns=False,
#             report_to="wandb",  # Using wandb for logging
#             seed=SEED,
#         )
        
#         # Initialize the standard Trainer instead of SFTTrainer
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             data_collator=collate_fn,
#             peft_config=peft_config
#         )
        
#         # Training
#         logger.info("*** Training ***")
#         checkpoint = None
#         if last_checkpoint is not None:
#             checkpoint = last_checkpoint
        
#         train_result = trainer.train(resume_from_checkpoint=checkpoint)
#         metrics = train_result.metrics
#         metrics["train_samples"] = len(train_dataset)
#         metrics["eval_samples"] = len(eval_dataset)
#         trainer.log_metrics("train", metrics)
#         trainer.save_metrics("train", metrics)
#         trainer.save_state()
        
#         # Save model
#         logger.info("*** Save model ***")
#         trainer.save_model(OUTPUT_DIR)
#         logger.info(f"Model saved to {OUTPUT_DIR}")
        
#         # Restore k,v cache for fast inference
#         model.config.use_cache = True
#         model.config.save_pretrained(OUTPUT_DIR)
        
#         # Make sure changes are persisted
#         runs_volume.commit()
        
#         logger.info("Training completed successfully.")
    
#     # Run the main training function
#     training()
@app.function(
    image=GPU_IMAGE,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
)
def train():
    import logging
    import os
    import sys
    import json
    from PIL import Image
    from typing import Optional

    import datasets
    import torch
    from torch.utils.data import Dataset
    import transformers
    from transformers import AutoProcessor, set_seed, Qwen2_5_VLForConditionalGeneration
    from transformers.trainer_utils import get_last_checkpoint
    from transformers import TrainingArguments
    from peft import LoraConfig
    from trl import SFTTrainer
    from datasets import Dataset as HFDataset
    os.environ["WANDB_PROJECT"] = "Qwen2.5VL-finetune"
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Hard-coded configuration
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    PROCESSED_DATA_PATH = "/data/processed_dataset"
    OUTPUT_DIR = "/runs/Qwen2.5-VL-vivid-layout"
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 1
    MAX_SEQ_LENGTH = 4096
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_GRADIENT_CHECKPOINTING = True
    USE_BF16 = True
    LOGGING_STEPS = 5
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 100
    SEED = 42

    # Set seed for reproducibility
    set_seed(SEED)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the process_vision_info function that was imported from qwen_vl_utils
    def process_vision_info(messages):
        """Process vision information from messages."""
        images = []
        videos = []
        
        for message in messages:
            if not isinstance(message.get("content", ""), list):
                continue
            
            for content in message["content"]:
                if not isinstance(content, dict):
                    continue
                    
                if content.get("type") == "image" and "image" in content:
                    image_path = content["image"]
                    if image_path.startswith("file://"):
                        image_path = image_path[7:]
                    
                    try:
                        img = Image.open(image_path)
                        images.append(img)
                    except Exception as e:
                        logger.error(f"Error loading image {image_path}: {e}")
                
                elif content.get("type") == "video" and "video" in content:
                    # Video processing if needed
                    videos.append(content["video"])
        
        return images, videos

    def load_dataset_to_hf_format(processed_data_path, split="train"):
        """Load the processed data and convert to HuggingFace dataset format."""
        import copy
        
        # Load the processed data
        if split == "train":
            processed_data_file = os.path.join(processed_data_path, "train_data.json")
        else:
            processed_data_file = os.path.join(processed_data_path, "eval_data.json")
        
        if not os.path.exists(processed_data_file):
            raise FileNotFoundError(f"Processed data file not found at {processed_data_file}. Run data_prep first.")
        
        with open(processed_data_file, "r") as f:
            processed_data = json.load(f)
                
        logger.info(f"Loaded {len(processed_data)} processed samples for {split} from {processed_data_file}")
        
        # Pre-process the data to make it PyArrow-friendly
        normalized_data = []
        for item in processed_data:
            # Create a normalized version of the item
            normalized_item = {
                # Store the image path as a string
                "image_path": item["image"],
                # Store the problem as a string
                "problem": item["problem"],
                # Convert messages to a serialized JSON string to avoid mixed types
                "messages_json": json.dumps(item["messages"])
            }
            normalized_data.append(normalized_item)
        
        logger.info(f"Normalized {len(normalized_data)} samples for PyArrow compatibility")
        
        # Convert to HuggingFace dataset format
        # This adds the column_names attribute that SFTTrainer expects
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_list(normalized_data)
        
        logger.info(f"Dataset columns: {hf_dataset.column_names}")
        
        # Load metadata
        metadata_file = os.path.join(processed_data_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            logger.info(f"Dataset metadata: {metadata}")
        
        return hf_dataset


    def formatting_func(example):
        """
        Convert from the normalized HF dataset format back to the original format
        needed for processing in the SFTTrainer.
        """
        # Parse the serialized messages back to their original structure
        messages = json.loads(example["messages_json"])
        
        # Return in the format expected by the model training code
        return {
            "messages": messages,
            # Include other fields if needed by the collator
            "image_path": example["image_path"]
        }

    def training():
        logger.info("*** Initializing processor ***")
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        logger.info("Using AutoProcessor for vision-language model.")
        
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
        elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        logger.info("*** Loading processed dataset ***")
        # Convert to HuggingFace datasets that work with SFTTrainer
        train_dataset = load_dataset_to_hf_format(PROCESSED_DATA_PATH, split="train")
        eval_dataset = load_dataset_to_hf_format(PROCESSED_DATA_PATH, split="eval")
        
        # Check for last checkpoint
        last_checkpoint = None
        if os.path.isdir(OUTPUT_DIR):
            last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
        
        logger.info("*** Initializing model ***")
        torch_dtype = torch.bfloat16 if USE_BF16 else torch.float32
        
        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_cache=False if USE_GRADIENT_CHECKPOINTING else True,
        )
        
        # Load the model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, **model_kwargs
        )
        
        # Define data formatting function
        def formatting_func(example):
            # SFTTrainer will handle this differently, we need to
            # structure the data in a way it can process
            return {
                "messages": example["messages"]
            }
        
        # Define data collator function
        def collate_fn(examples):
            # Extract messages from each example after they've been processed by formatting_func
            message_lists = [example["messages"] for example in examples]
            
            # Apply chat template to get the text
            texts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in message_lists
            ]
            
            # Process images for each example
            image_inputs = []
            for messages in message_lists:
                imgs, vids = process_vision_info(messages)
                image_inputs.append(imgs)
            
            # Create the batch with processor
            batch = processor(
                text=texts,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
            )
            
            # Create the labels by cloning input_ids and masking appropriately
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            # Handle image token ID if available
            if hasattr(processor, "image_token"):
                image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
                labels[labels == image_token_id] = -100
            
            batch["labels"] = labels
            return batch
            
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
            bf16=USE_BF16,
            logging_steps=LOGGING_STEPS,
            evaluation_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=EVAL_STEPS,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="wandb",  # Using wandb for logging
            seed=SEED,
        )
        
        # Initialize SFTTrainer - now should work with our HF dataset format
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            formatting_func=formatting_func,
            peft_config=peft_config,
            # Prevent automatic masking of inputs based on format detection
            # as we handle this manually in our collate_fn
            # dataset_text_field=None,
            # Stop SFTTrainer from trying to apply tokenizer directly
            # tokenizer=None,  
            # max_seq_length=MAX_SEQ_LENGTH
        )
        
        # Training
        logger.info("*** Training ***")
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save model
        logger.info("*** Save model ***")
        trainer.save_model(OUTPUT_DIR)
        logger.info(f"Model saved to {OUTPUT_DIR}")
        
        # Restore k,v cache for fast inference
        model.config.use_cache = True
        model.config.save_pretrained(OUTPUT_DIR)
        
        # Make sure changes are persisted
        runs_volume.commit()
        
        logger.info("Training completed successfully.")
    
    # Run the main training function
    training()


@app.function(
    image=SWIFT_GPU_IMAGE,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
)
def train_swift():
    import logging
    import os
    import sys
    import subprocess
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define paths
    DATASET_PATH = "/data/swift_output"
    OUTPUT_DIR = "/runs/swift_output"
    
    # Check if the dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run data_prep_swift first.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Validate dataset before starting training
    logger.info("Validating dataset...")
    combined_jsonl_path = os.path.join(DATASET_PATH, "transformed_data.jsonl")
    if not os.path.exists(combined_jsonl_path):
        raise FileNotFoundError(f"Dataset file not found at {combined_jsonl_path}")
    
    # Read a few examples to validate format
    valid_count = 0
    invalid_count = 0
    with open(combined_jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Just check the first 10 examples
                break
            
            try:
                example = json.loads(line.strip())
                
                # Validate messages field
                if not example.get("messages") or not isinstance(example["messages"], list) or len(example["messages"]) == 0:
                    logger.warning(f"Example {i} has invalid messages field: {example.get('messages')}")
                    invalid_count += 1
                    continue
                
                # Validate images field
                if not example.get("images") or not isinstance(example["images"], list) or len(example["images"]) == 0:
                    logger.warning(f"Example {i} has invalid images field: {example.get('images')}")
                    invalid_count += 1
                    continue
                
                # Check if image files exist
                for img_path in example["images"]:
                    if not os.path.exists(img_path):
                        logger.warning(f"Image file not found: {img_path}")
                        invalid_count += 1
                        continue
                
                valid_count += 1
            
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON at line {i}")
                invalid_count += 1
    
    logger.info(f"Dataset validation: {valid_count} valid examples, {invalid_count} invalid examples")
    
    # Check if cuda is available
    try:
        import torch
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not available, can't check CUDA status")
    
    # Print environment for debugging
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if key.startswith("CUDA") or key.startswith("SWIFT") or key.startswith("HF_"):
            logger.info(f"  {key}={value}")
    
    # Set environment variables
    os.environ["MAX_PIXELS"] = "1003520"
    
    # Define the Swift command with updated parameters based on your last run
    swift_command = [
        "swift", "sft",
        "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
        "--dataset", DATASET_PATH,
        "--train_type", "lora",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--learning_rate", "1e-4",
        "--lora_rank", "16",  # Updated from 8 to 16
        "--lora_alpha", "32",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--gradient_accumulation_steps", "16",
        "--eval_steps", "100",
        "--save_steps", "100",
        "--save_total_limit", "5",
        "--logging_steps", "5",
        "--max_length", "4048",  # Updated from 2048 to 4048
        "--output_dir", OUTPUT_DIR,
        "--warmup_ratio", "0.05",
        "--dataloader_num_workers", "4",
        "--dataset_num_proc", "4",
        "--use_hf", "true",
        "--ignore_data_skip"  # Add option to skip invalid data
    ]
    
    # Log the command
    logger.info(f"Running Swift command: {' '.join(swift_command)}")
    
    # Run the Swift command
    try:
        # Run the process and stream output in real-time
        process = subprocess.Popen(
            swift_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for the process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Swift training completed successfully")
        else:
            logger.error(f"Swift training failed with return code {return_code}")
            raise subprocess.CalledProcessError(return_code, swift_command)
    
    except Exception as e:
        logger.error(f"Error during Swift training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Make sure changes are persisted
        runs_volume.commit()
    
    return {"status": "completed", "output_dir": OUTPUT_DIR}
# def train_swift():
#     import logging
#     import os
#     import sys
#     import subprocess
    
#     # Configure logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)

#     # Define paths
#     DATASET_PATH = "/data/swift_output"
#     OUTPUT_DIR = "/runs/swift_output"
    
#     # Check if the dataset exists
#     if not os.path.exists(DATASET_PATH):
#         raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run data_prep_swift first.")
    
#     # Create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Check if cuda is available
#     try:
#         import torch
#         logger.info(f"CUDA available: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             logger.info(f"CUDA device count: {torch.cuda.device_count()}")
#             logger.info(f"CUDA current device: {torch.cuda.current_device()}")
#             logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
#     except ImportError:
#         logger.warning("PyTorch not available, can't check CUDA status")
    
#     # Print environment for debugging
#     logger.info("Environment variables:")
#     for key, value in os.environ.items():
#         if key.startswith("CUDA") or key.startswith("SWIFT") or key.startswith("HF_"):
#             logger.info(f"  {key}={value}")
    
#     # Set environment variables
#     os.environ["MAX_PIXELS"] = "1003520"
    
#     # Define the Swift command
#     swift_command = [
#         "swift", "sft",
#         "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
#         "--dataset", DATASET_PATH,
#         "--train_type", "lora",
#         "--torch_dtype", "bfloat16",
#         "--num_train_epochs", "1",
#         "--per_device_train_batch_size", "1",
#         "--per_device_eval_batch_size", "1",
#         "--learning_rate", "1e-4",
#         "--lora_rank", "16",
#         "--lora_alpha", "32",
#         "--target_modules", "all-linear",
#         "--freeze_vit", "true",
#         "--gradient_accumulation_steps", "16",
#         "--eval_steps", "100",
#         "--save_steps", "100",
#         "--save_total_limit", "5",
#         "--logging_steps", "5",
#         "--max_length", "4048",
#         "--output_dir", OUTPUT_DIR,
#         "--warmup_ratio", "0.05",
#         "--dataloader_num_workers", "4",
#         "--dataset_num_proc", "4",
#         "--use_hf", "true",
#         "--ignore_data_skip"
#     ]
    
#     # Log the command
#     logger.info(f"Running Swift command: {' '.join(swift_command)}")
    
#     # Run the Swift command
#     try:
#         # Run the process and stream output in real-time
#         process = subprocess.Popen(
#             swift_command, 
#             stdout=subprocess.PIPE, 
#             stderr=subprocess.STDOUT,
#             universal_newlines=True,
#             bufsize=1
#         )
        
#         # Stream the output
#         for line in process.stdout:
#             logger.info(line.strip())
        
#         # Wait for the process to complete
#         return_code = process.wait()
        
#         if return_code == 0:
#             logger.info("Swift training completed successfully")
#         else:
#             logger.error(f"Swift training failed with return code {return_code}")
#             raise subprocess.CalledProcessError(return_code, swift_command)
    
#     except Exception as e:
#         logger.error(f"Error during Swift training: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise
#     finally:
#         # Make sure changes are persisted
#         runs_volume.commit()
    
#     return {"status": "completed", "output_dir": OUTPUT_DIR}