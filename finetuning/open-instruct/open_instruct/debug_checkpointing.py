import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType


# Define arguments
class Args:
    def __init__(self):
        self.model_name_or_path = "gpt2"
        self.resume_from_checkpoint = (
            "./checkpoint"  # None  # Set to checkpoint path to resume training
        )
        self.output_dir = "./checkpoint"
        self.use_lora = True
        self.adapter_name = "default"
        self.learning_rate = 1e-4
        self.num_training_steps = 1000
        self.num_warmup_steps = 100
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.deepspeed_config = "ds_config.json"


args = Args()

# Initialize Accelerator
accelerator = Accelerator()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# Configure LoRa if enabled
if args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["attn.c_attn", "mlp.c_proj"],
    )
    model = get_peft_model(model, peft_config)

# Prepare optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=args.num_warmup_steps
)

# Prepare model, optimizer, and scheduler with Accelerator
model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)


# Function to save the checkpoint using DeepSpeed
def save_checkpoint(accelerator, model, tokenizer, optimizer, scheduler, output_dir):
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)

    if optimizer is not None:
        optimizer_path = os.path.join(output_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)

    if scheduler is not None:
        scheduler_path = os.path.join(output_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_path)

    accelerator_state_path = os.path.join(output_dir, "accelerator_state.pt")
    # Save accelerator state
    with open(accelerator_state_path, "wb") as f:
        torch.save(accelerator.get_state_dict(model), f)

    tokenizer.save_pretrained(output_dir)


# Function to load the checkpoint using DeepSpeed
def load_checkpoint(accelerator, model, optimizer, scheduler, args):
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

        if args.use_lora:
            model.load_adapter(checkpoint_path, adapter_name=args.adapter_name)
        else:
            model.load_state_dict(
                torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
            )

        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path) and optimizer is not None:
            optimizer.load_state_dict(torch.load(optimizer_path))

        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(scheduler_path) and scheduler is not None:
            scheduler.load_state_dict(torch.load(scheduler_path))

        accelerator_state_path = os.path.join(checkpoint_path, "accelerator_state.pt")
        # Load accelerator state
        with open(accelerator_state_path, "rb") as f:
            state = torch.load(f)
        accelerator.load_state_dict(state)


# Training loop (simplified)
def train():
    model.train()  # Ensure the model is in training mode
    for step in range(args.num_training_steps):
        # Dummy input data for the example
        input_ids = (
            torch.tensor(tokenizer.encode("Hello, world!", add_special_tokens=True))
            .unsqueeze(0)
            .to(accelerator.device)
        )
        labels = input_ids.clone().to(accelerator.device)

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        accelerator.backward(loss)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            save_checkpoint(
                accelerator, model, tokenizer, optimizer, scheduler, args.output_dir
            )
            print(f"Checkpoint saved at step {step}")


# Load checkpoint if specified
if args.resume_from_checkpoint:
    load_checkpoint(accelerator, model, optimizer, scheduler, args)

# Start training
train()
