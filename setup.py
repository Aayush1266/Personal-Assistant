# setup.py
import os
import time
import torch
import pickle
import tiktoken

from GPTModel import GPTModel
from data_utils import load_data
from utility_functions import evaluate_model, calc_loss_batch, calc_accuracy_loader,format_input,train_model_simple
from gpt_download import download_and_load_gpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== MODEL-CONFIGURATION ====================
CHOSSE_MODEL = "gpt2-small (124M)"
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
BASE_CONFIG.update(model_configs[CHOSSE_MODEL])


# ==================== DATA LOADING ====================
tokenizer = tiktoken.get_encoding("gpt2")
train_loader, val_loader, test_loader, max_length, val_data= load_data(
    tokenizer=tokenizer, batch_size=8,
)


# ==================== MODEL INIT ====================
model_size = CHOSSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# settings, params = download_and_load_gpt(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
# load_weights_into_gpt(model, params)


# ==================== FREEZE & MODIFY LAST LAYERS ====================
for param in model.parameters():
    param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

# Last block and final norm trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

model = model.to(device)

# ==================== TRAINING FUNCTION ====================
def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradient from privious batch
            loss = calc_loss_batch(input_batch, target_batch, model, device) # calculate loss
            loss.backward() # calculate loss gradients through backward propogation
            optimizer.step() # update weights

            examples_seen += input_batch.shape[0]
            global_step += 1

            # optional
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): Train loss={train_loss:.3f}, Val loss={val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print(f"Train Acc: {train_accuracy*100:.2f}%, Val Acc: {val_accuracy*100:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# ==================== FINE-TUNING ====================
def finetune(num_epochs=5):
    PATH = "gpt2-small124M-sft.pth"
    # stats_path = "classification-stats.pkl"

    if os.path.exists(PATH):
        print(f"{PATH} already exists. Skipping training!")
        return

    # Load data
    train_loader, val_loader, test_loader,max_length, val_data= load_data(
        tokenizer=tiktoken.get_encoding("gpt2"), batch_size=8,
    )

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    print("Training starting...")

    import time

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 1
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

finetune()