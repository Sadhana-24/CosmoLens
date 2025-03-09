import os
from modal import App, Image, Volume, Secret
from pathlib import Path

# Create the Modal app
app = App("qwen2-vl-fine-tuning")

# Create secrets for API keys
huggingface_secret = Secret.from_name("huggingface-secret")
wandb_secret = Secret.from_name("wandb-secret")

# Create volumes for persistent storage
model_volume = Volume.from_name("qwen2-models-vol", create_if_missing=True)
runs_volume = Volume.from_name("qwen2-runs-vol", create_if_missing=True)
data_volume = Volume.from_name("qwen2-data-vol", create_if_missing=True)

# Configure volume mounting points
VOLUME_CONFIG = {
    "/pretrained": model_volume,
    "/runs": runs_volume,
    "/data": data_volume,
}

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

# Time constants
HOURS = 60 * 60
MINUTES = 60

# Environment variables
GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100-80gb:2")
if len(GPU_CONFIG.split(":")) <= 1:
    N_GPUS = int(os.environ.get("N_GPUS", 1))
    GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"
SINGLE_GPU_CONFIG = os.environ.get("SINGLE_GPU_CONFIG", "a100-80gb:1")

@app.function(
    image=GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
    secrets=[huggingface_secret, wandb_secret, Secret.from_dotenv()],
    gpu=GPU_CONFIG,
)
def fine_tune_qwen2_vl():
    import logging
    import sys
    import torch
    from datasets import load_dataset
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    import wandb
    from qwen_vl_utils import process_vision_info

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Load dataset
    dataset_id = "HuggingFaceM4/ChartQA"
    train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])

    # Format dataset
    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


    def format_data(sample):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample['query'],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["label"][0]
                    }
                ],
            },
        ]

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    # Load model and processor
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="/runs/qwen2-7b-instruct-trl-sft-ChartQA",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=True,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.remove_unused_columns = False

    # Initialize W&B
    wandb.init(
        project="qwen2-7b-instruct-trl-sft-ChartQA",
        name="qwen2-7b-instruct-trl-sft-ChartQA",
        config=training_args,
    )

    from huggingface_hub import login
    login(token='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')

    # Define collator function
    def collate_fn(examples):
        texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example)[0] for example in examples]
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)

    # Log the model to W&B
    wandb.finish()

if __name__ == "__main__":
    fine_tune_qwen2_vl.remote()