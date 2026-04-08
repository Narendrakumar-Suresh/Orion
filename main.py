import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from blocks.orion import Orion
from upload import upload

# --- 1. Config ---
class Config:
    vocab_size   = 128001  
    hidden_dim   = 512
    num_layers   = 12
    num_heads    = 8
    ffn_dim      = 2048
    z_dim        = 256
    max_seq_len  = 128

    batch_size   = 128
    para_len     = 8
    lr           = 5e-5
    lambda_i     = 0.1
    temp         = 0.07
    total_steps  = 37536

cfg = Config()

# --- 2. Data Streaming ---
def get_hf_stream_batches(repo_id, cfg):
    ds = load_dataset(repo_id, split="train", streaming=True)
    tokens_per_batch = cfg.batch_size * cfg.para_len * cfg.max_seq_len
    buffer = []
    
    for row in ds:
        buffer.extend(row['input_ids'])
        while len(buffer) >= tokens_per_batch:
            batch_tokens = buffer[:tokens_per_batch]
            buffer = buffer[tokens_per_batch:]
            
            data = torch.tensor(batch_tokens, dtype=torch.long)
            yield data.view(cfg.batch_size, cfg.para_len, cfg.max_seq_len)

# --- 3. Loss Function ---
def compute_loss(model, paragraph, z_init, cfg):
    B, T_para, T_seq = paragraph.shape
    z = z_init
    total_nll = 0.
    total_intent = 0.
    labels = torch.arange(B, device=paragraph.device)

    for t in range(T_para - 1):
        tokens_t = paragraph[:, t, :]
        tokens_next = paragraph[:, t+1, :]

        # Forward pass
        logits, z_t = model(tokens_t, z)
        
        # Get target z_next without tracking gradients
        with torch.no_grad():
            _, z_next = model(tokens_next, z_t)

        # Token NLL Loss
        nll = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, cfg.vocab_size), 
            tokens_t[:, 1:].reshape(-1)
        )

        # Intent Contrastive Loss
        z_t_n = F.normalize(z_t, p=2, dim=-1)
        z_next_n = F.normalize(z_next, p=2, dim=-1)
        sim_matrix = torch.matmul(z_t_n, z_next_n.T) / cfg.temp
        loss_intent = F.cross_entropy(sim_matrix, labels)

        total_nll += nll
        total_intent += loss_intent

        # Stop gradient for the next paragraph step
        z = z_t.detach()

    steps = T_para - 1
    l_nll = total_nll / steps
    l_int = total_intent / steps
    total_loss = l_nll + (cfg.lambda_i * l_int)
    
    return total_loss, l_nll, l_int

# --- 4. Training Loop ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("naren-1219/orion-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    # Setup Model & Compile
    model = Orion(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    model = torch.compile(model) 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    # ---------------------------------------------------------
    # SILENTLY CALCULATE STEPS FOR THE SCHEDULER
    # This prevents the scheduler from crashing, but doesn't force a hard stop
    target_tokens = 300_000_000
    tokens_per_step = cfg.batch_size * cfg.para_len * cfg.max_seq_len
    estimated_steps = target_tokens // tokens_per_step
    scheduler = CosineAnnealingLR(optimizer, T_max=estimated_steps)
    # ---------------------------------------------------------

    batch_gen = get_hf_stream_batches("naren-1219/fineweb-edu-10bt-tokenized", cfg)

    model.train()
    
    # Just loop over the generator. When the dataset is empty, the loop ends automatically!
    for step, batch in enumerate(batch_gen):

        paragraph = batch.to(device)
        z_init = torch.zeros(cfg.batch_size, cfg.z_dim, device=device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, nll, intent = compute_loss(model, paragraph, z_init, cfg)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step ==1:
            print(f"Step {step} | Loss: {loss.item():.4f} | NLL: {nll.item():.4f} | Intent: {intent.item():.4f}")
        
        if step ==2:
            print(f"Step {step} | Loss: {loss.item():.4f} | NLL: {nll.item():.4f} | Intent: {intent.item():.4f}")

        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | NLL: {nll.item():.4f} | Intent: {intent.item():.4f}")

        if step % 500 == 0 and step > 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                }, f"orion_step_{step}.pt")

            print(f"Checkpoint saved at step {step}")

    # This runs exactly when the dataset stream runs out of data
    torch.save(model.state_dict(), "orion_final.pt")
    print("Dataset empty. Training Complete!")

if __name__ == "__main__":
    train()
    upload()