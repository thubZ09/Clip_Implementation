import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer
from PIL import Image
from typing import cast


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

def find_matches(model, image_path: str, captions: list, device="cuda"):
    model.eval()
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    pil_image = Image.open(image_path).convert("RGB")
    
    img_tensor = cast(torch.Tensor, transform(pil_image))
    img_tensor = img_tensor.unsqueeze(0)  
    img = img_tensor.to(device)
    
    text = tokenizer(captions, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        img_emb, txt_emb, scale = model({
            "images": img, 
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"]
        })
    
    scores = (img_emb @ txt_emb.t()) * scale
    return torch.topk(scores, k=5, dim=1)


