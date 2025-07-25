import os
from re import split
import subprocess
from modal import App, Image, Volume, Secret
from pathlib import Path
from datasets import load_dataset
from common import VOLUME_CONFIG, CPU_IMAGE, SWIFT_GPU_IMAGE, HOURS
from common import SINGLE_GPU_CONFIG  
from common import data_volume, runs_volume, model_volume
import multiprocessing

# Create the Modal app
app = App("galaxy-classification-trainer")
huggingface_secret = Secret.from_name("huggingface-secret")
wandb_secret = Secret.from_name("wandb-secret")

@app.function(
    image=CPU_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=12 * HOURS,
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

    # Define paths for ESA Hubble dataset
    DATASET_NAME = "Supermaxman/esa-hubble"
    OUTPUT_DIR = "/data/esa_hubble_data"
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    JSONL_DIR = os.path.join(OUTPUT_DIR, "jsonl")

    # Create output directories
    dirs_to_create = [OUTPUT_DIR, IMAGE_DIR, JSONL_DIR]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    logger.info(f"Loading dataset {DATASET_NAME}")
    # Load the ESA Hubble dataset
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Sample a portion if needed
    SAMPLE_SIZE = 1000  # Limit to 1,000 examples
    dataset = dataset.take(SAMPLE_SIZE)  # Use `take` to limit the dataset size

    logger.info(f"Loaded {SAMPLE_SIZE} examples")

    def generate_qa_pairs(item):
        """Generate multi-task question-answer pairs from ESA Hubble dataset."""
        # Extract image and metadata
        image = item["image"]
        title = item["title"]
        description = item["description"]
        metadata = {
            "id": item["id"],
            "credits": item["credits"],
            "url": item["url"],
            "release_date": item["Release date"],
            "constellation": item["Constellation"],
            "distance": item["Distance"],
            "category": item["Category"],
        }

        # Generate question-answer pairs
        qa_pairs = []

        # Task 1: Image Captioning
        qa_pairs.append({
            "question": "Describe the image in detail.",
            "answer": description
        })

        # Task 2: Object Identification
        qa_pairs.append({
            "question": "What celestial objects are present in the image?",
            "answer": title  # Use the title as a summary of objects
        })

        # Task 3: Metadata Query
        qa_pairs.append({
            "question": "What is the distance to the celestial object in the image?",
            "answer": metadata["distance"]
        })

        # Task 4: Constellation Identification
        qa_pairs.append({
            "question": "Which constellation does the celestial object belong to?",
            "answer": metadata["constellation"]
        })

        # Task 5: Scientific Context
        qa_pairs.append({
            "question": "What scientific significance does this image hold?",
            "answer": description  # Use the description for context
        })

        return qa_pairs, image, metadata

    def process_single_item(item, idx, pbar=None):
        """Process a single dataset item for ESA Hubble dataset."""
        try:
            # Generate QA pairs
            qa_pairs, image, metadata = generate_qa_pairs(item)
    
            # Save the image
            image_filename = f"hubble_{idx}.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            image.save(image_path)
    
            # Convert image path to Unix style for consistency
            unix_style_path = image_path.replace(os.sep, '/')
    
            # Create the Swift-compatible format with all QA pairs
            transformed_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant specializing in astronomy. Your task is to answer questions about deep space images captured by the Hubble Space Telescope."
                    }
                ],
                "images": [unix_style_path],
                "metadata": metadata
            }
    
            # Add all QA pairs to the messages
            for qa_pair in qa_pairs:
                transformed_item["messages"].append({
                    "role": "user",
                    "content": f"<image> {qa_pair['question']}"
                })
                transformed_item["messages"].append({
                    "role": "assistant",
                    "content": qa_pair["answer"]
                })
    
            # Save individual JSONL file
            jsonl_path = os.path.join(JSONL_DIR, f"hubble_{idx}.jsonl")
            with open(jsonl_path, 'w') as f:
                f.write(json.dumps(transformed_item))
    
            result = {
                "item": transformed_item,
                "id": idx,
                "qa_pairs": qa_pairs
            }
    
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            result = None
        finally:
            if pbar is not None:
                pbar.update(1)
    
        return result

    # Process the dataset
    logger.info(f"Processing {SAMPLE_SIZE} items")
    transformed_data = []

    # Use ThreadPoolExecutor for parallel processing
    with tqdm(total=SAMPLE_SIZE, desc="Processing with ThreadPoolExecutor") as pbar:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {}
            idx = 0
            for item in dataset:
                futures[executor.submit(process_single_item, item, idx, pbar)] = idx
                idx += 1

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        transformed_data.append(result["item"])
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Error processing item {idx}: {e}")

    # Split into train/val sets
    import random
    random.shuffle(transformed_data)
    split_idx = int(len(transformed_data) * 0.9)  # 90% train, 10% val
    train_data = transformed_data[:split_idx]
    val_data = transformed_data[split_idx:]

    # Save train data as single JSONL file
    train_jsonl_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    logger.info(f"Saving train data ({len(train_data)} examples) to {train_jsonl_path}")
    with open(train_jsonl_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    # Save validation data as single JSONL file
    val_jsonl_path = os.path.join(OUTPUT_DIR, "val.jsonl")
    logger.info(f"Saving validation data ({len(val_data)} examples) to {val_jsonl_path}")
    with open(val_jsonl_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    # Create dataset info
    dataset_info = {
        "dataset_name": DATASET_NAME,
        "num_samples": len(transformed_data),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "output_dir": OUTPUT_DIR,
        "train_jsonl_path": train_jsonl_path,
        "val_jsonl_path": val_jsonl_path,
    }

    # Save dataset info
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Successfully processed {len(transformed_data)} examples")
    logger.info(f"Dataset prepared for Swift at {OUTPUT_DIR}")

    # Make sure changes are persisted
    data_volume.commit()

    return {"samples_processed": len(transformed_data), "output_dir": OUTPUT_DIR}


@app.function(
    image=SWIFT_GPU_IMAGE,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,  # Extended for training on more data
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
    os.environ["WANDB_PROJECT"] = "GalaxyClassification-finetune"
    
    # Define paths
    DATASET_PATH = "/data/swift_galaxy_data_new/train.jsonl"
    VAL_DATASET_PATH = "/data/swift_galaxy_data_new/val.jsonl"
    OUTPUT_DIR = "/runs/galaxy_classification_output_new"
    
    # Check if the dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run data_prep_swift first.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set environment variables
    os.environ["MAX_PIXELS"] = "1003520"
    
    # Define the Swift command with parameters optimized for galaxy classification
    swift_command = [
    "swift", "sft",
    "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
    # "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
    # "--model", "Qwen/Qwen2-VL-2B-Instruct",
    # "--model", "AdithyaSK/ViViDLayout2b",
    "--dataset", DATASET_PATH,
    "--train_type", "lora",
    # "--train_type", "full",
    "--torch_dtype", "bfloat16",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--learning_rate", "1e-4",
    "--lora_rank", "16",  # Updated from 8 to 16
    "--lora_alpha", "32",
    "--target_modules", "all-linear",
    "--freeze_vit", "true",
    "--gradient_accumulation_steps", "4",
    "--eval_steps", "100",
    "--save_steps", "100",
    "--save_total_limit", "5",
    "--logging_steps", "10",
    "--max_length", "4048",  # Updated from 2048 to 4048
    "--output_dir", OUTPUT_DIR,
    "--warmup_ratio", "0.05",
    "--dataloader_num_workers", "4",
    "--dataset_num_proc", "4",
    "--use_hf", "true",
    "--report_to", "wandb",
    "--hub_model_id", "Samarth0710/Test_2_5VL_7B"
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



@app.function(
    image=SWIFT_GPU_IMAGE,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=12 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
)
def push_swift():
    """
    Export and push the fine-tuned Swift model to Hugging Face Hub.
    This function converts the LoRA adapter to a full model and uploads it.
    """
    import logging
    import os
    import sys
    import subprocess
    import json

    # pip install verovio.
    subprocess.run(["pip", "install", "verovio"])
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Define paths
    # MODEL_PATH = "/runs/swift_output_update/v5-20250310-105440/checkpoint-512"  # Path to the trained model
    # MODEL_PATH = "/runs/swift_output_update_3b/v1-20250310-135133/checkpoint-512"  # Path to the trained model
    # MODEL_PATH = "/runs/swift_output_update_2b/v0-20250312-064836/checkpoint-512"
    # MODEL_PATH = "/runs/swift_output_update_0.5b_got/v1-20250310-142429/checkpoint-512"
    MODEL_PATH = "/runs/galaxy_classification_output/v0-20250312-184507/checkpoint-1800"  # Path to the trained model
    
    # Use a timestamp to avoid conflicts with existing directories
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # EXPORTED_MODEL_PATH = f"/runs/swift_exported_model_{time"  # Path for the exported model
    
    HUB_MODEL_ID = "Samarth0710/gz10_classifier_7B"  # Your HF hub model ID
    
    # # Handle existing output directory - Swift requires the directory to not exist
    # if os.path.exists(EXPORTED_MODEL_PATH):
    #     import shutil
    #     logger.info(f"Output directory {EXPORTED_MODEL_PATH} already exists. Removing it...")
    #     shutil.rmtree(EXPORTED_MODEL_PATH)
    
    # # Create fresh output directory
    # os.makedirs(EXPORTED_MODEL_PATH)
    
    # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train model first.")
    
    # Get Hugging Face token from environment
    hf_token = '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'
    if not hf_token:
        logger.error("HUGGINGFACE_TOKEN environment variable not found")
        raise ValueError("Hugging Face token not available")
    
    # Define the Swift export command
    swift_command = [
        "swift", "export",
        "--adapters", MODEL_PATH,
        "--merge_lora", "true",
        "--push_to_hub", "true",
        "--hub_model_id", HUB_MODEL_ID,
        "--use_hf", "true",  # Added the use_hf parameter
        "--hub_token", hf_token,
    ]
    
    # Log the command (hide token)
    log_command = swift_command.copy()
    log_command[-1] = "****"  # Hide actual token in logs
    logger.info(f"Running Swift export command: {' '.join(log_command)}")
    
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
            # Skip logging lines that might contain the token
            if hf_token not in line:
                logger.info(line.strip())
        
        # Wait for the process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Swift model export and push completed successfully")
        else:
            logger.error(f"Swift model export failed with return code {return_code}")
            raise subprocess.CalledProcessError(return_code, log_command)
    
    except Exception as e:
        logger.error(f"Error during Swift model export: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Make sure changes are persisted
        runs_volume.commit()
    
    # Create model card with information
    try:
        model_card_content = f"""
# gz10 Classifier 

This model is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for Galaxy classifiaction.

## Model Details

- **Base Model:** Qwen/Qwen2.5-VL-3B-Instruct
- **Training Framework:** MS-Swift
- **Fine-tuning Technique:** LoRA (rank={16}, alpha={32})
- **Target Modules:** all-linear


"""
        
        # Write model card to the repository
        # with open(os.path.join(EXPORTED_MODEL_PATH, "README.md"), "w") as f:
        #     f.write(model_card_content)
        
        logger.info("Created model card README.md")
    except Exception as e:
        logger.warning(f"Failed to create model card: {e}")
    
    return {
        "status": "completed", 
        "model_path": MODEL_PATH,
        "hub_model_id": HUB_MODEL_ID
    }




@app.function(
    image=SWIFT_GPU_IMAGE,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()]
)
def evaluate_models():
    import logging
    import os
    import json
    from datasets import load_dataset
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from sklearn.metrics import accuracy_score
    import torch

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define paths and model IDs
    DATASET_NAME = "MultimodalUniverse/gz10"
    BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    FINE_TUNED_MODEL_ID = "Samarth0710/gz10_classifier_7B"
    SAMPLE_SIZE = 100  # Evaluate on a small subset

    # Load a small subset of the dataset
    logger.info(f"Loading dataset {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train").shuffle(seed=42).select(range(SAMPLE_SIZE))

    # Load base model and processor
    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    base_processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    # Load fine-tuned model and processor
    logger.info(f"Loading fine-tuned model: {FINE_TUNED_MODEL_ID}")
    fine_tuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        FINE_TUNED_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    fine_tuned_processor = AutoProcessor.from_pretrained(FINE_TUNED_MODEL_ID)

    # Class labels mapping for clarity in prompts
    CLASS_LABELS = {
        0: "Disturbed Galaxies",
        1: "Merging Galaxies",
        2: "Round Smooth Galaxies", 
        3: "In-between Round Smooth Galaxies",
        4: "Cigar Shaped Smooth Galaxies",
        5: "Barred Spiral Galaxies",
        6: "Unbarred Tight Spiral Galaxies",
        7: "Unbarred Loose Spiral Galaxies",
        8: "Edge-on Galaxies without Bulge",
        9: "Edge-on Galaxies with Bulge"
    }

    # Prompts for galaxy classification task
    SYSTEM_PROMPT = "You are an AI assistant specializing in galaxy morphology classification. Your task is to classify galaxies based on their visual appearance into different morphological types."
    USER_PROMPT = "Please classify this galaxy image into one of the following categories: Disturbed Galaxies, Merging Galaxies, Round Smooth Galaxies, In-between Round Smooth Galaxies, Cigar Shaped Smooth Galaxies, Barred Spiral Galaxies, Unbarred Tight Spiral Galaxies, Unbarred Loose Spiral Galaxies, Edge-on Galaxies without Bulge, Edge-on Galaxies with Bulge."

    def predict(model, processor, item):
        """Run inference on a single item using the given model."""
        try:
            # Prepare the input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item["rgb_image"],  # Use the image from the dataset
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }
            ]

            # Prepare inputs for the model
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # Extract the predicted class from the output
            predicted_class = None
            for label_id, class_name in CLASS_LABELS.items():
                if class_name.lower() in output_text.lower():
                    predicted_class = label_id
                    break

            return predicted_class

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None

    # Evaluate both models
    logger.info("Evaluating base model...")
    base_predictions = []
    true_labels = []
    for item in dataset:
        true_label = item["gz10_label"]
        predicted_label = predict(base_model, base_processor, item)
        if predicted_label is not None:
            base_predictions.append(predicted_label)
            true_labels.append(true_label)

    base_accuracy = accuracy_score(true_labels, base_predictions)
    logger.info(f"Base model accuracy: {base_accuracy:.2f}")

    logger.info("Evaluating fine-tuned model...")
    fine_tuned_predictions = []
    true_labels = []
    for item in dataset:
        true_label = item["gz10_label"]
        predicted_label = predict(fine_tuned_model, fine_tuned_processor, item)
        if predicted_label is not None:
            fine_tuned_predictions.append(predicted_label)
            true_labels.append(true_label)

    fine_tuned_accuracy = accuracy_score(true_labels, fine_tuned_predictions)
    logger.info(f"Fine-tuned model accuracy: {fine_tuned_accuracy:.2f}")

    # Compare results
    results = {
        "base_model_accuracy": base_accuracy,
        "fine_tuned_model_accuracy": fine_tuned_accuracy,
        "improvement": fine_tuned_accuracy - base_accuracy,
        "sample_size": SAMPLE_SIZE
    }

    # Save results
    results_path = "/runs/evaluation_results_7B.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {results_path}")

    # Make sure changes are persisted
    runs_volume.commit()

    return results