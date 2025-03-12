
import torch
from transformers import ViTModel, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from configs import LocalConfig  

class LocalCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = LocalConfig()
        
        # Image encoder
        self.vis_encoder = ViTModel.from_pretrained(self.cfg.IMAGE_MODEL)
        self.vis_proj = nn.Linear(self.vis_encoder.config.hidden_size, self.cfg.PROJ_DIM)
        
        # Text encoder
        self.txt_encoder = AutoModel.from_pretrained(self.cfg.TEXT_MODEL)
        self.txt_proj = nn.Linear(self.txt_encoder.config.hidden_size, self.cfg.PROJ_DIM)
        
        self.logit_scale = nn.Parameter(torch.tensor(1/0.07).log())

    def forward(self, batch):
        # Image forward pass
        img_feats = self.vis_encoder(pixel_values=batch["images"]).last_hidden_state[:, 0]
        img_emb = F.normalize(self.vis_proj(img_feats), dim=-1)
        
        # Text forward pass
        txt_feats = self.txt_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        ).last_hidden_state[:, 0]
        txt_emb = F.normalize(self.txt_proj(txt_feats), dim=-1)
        
        return img_emb, txt_emb, self.logit_scale.exp()
