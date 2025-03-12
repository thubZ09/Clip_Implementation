
import torch
import torch.nn.functional as F
from torch.optim import AdamW 
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LambdaLR
from configs import LocalConfig
from model import LocalCLIP
from data_utils import get_dataloader

def train():
    cfg = LocalConfig()
    model = LocalCLIP().to(cfg.DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.LR)
    scaler = GradScaler(enabled=cfg.FP16)
    
    # Update these paths based on your local dataset location:
    csv_path = "./data/flickr-image-dataset/results.csv"
    img_dir = "./data/flickr-image-dataset"
    train_loader = get_dataloader(csv_path, img_dir, cfg.BATCH_SIZE)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(step / cfg.WARMUP, 1.0))
    
   
    device_type = "cuda" if cfg.DEVICE == "cuda" else "cpu"
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(cfg.DEVICE) for k, v in batch.items()}
            
            with autocast(enabled=cfg.FP16, device_type=device_type):
                img_emb, txt_emb, scale = model(batch)
                logits = scale * img_emb @ txt_emb.t()
                targets = torch.arange(len(logits), device=cfg.DEVICE)
                loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2
            
            scaler.scale(loss).backward()
            
            if (i+1) % cfg.GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            if i % 50 == 0:
                print(f"Epoch: {epoch+1}/{cfg.EPOCHS} | Step: {i} | Loss: {loss.item():.4f}")
                
            if i % 200 == 0 and cfg.SAVE_PATH:
                checkpoint_path = f"{cfg.SAVE_PATH}clip_epoch{epoch}_step{i}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
