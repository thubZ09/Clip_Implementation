
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, BatchEncoding
import pandas as pd
from PIL import Image
from configs import LocalConfig

class Flickr30kDataset(Dataset):
    def __init__(self, csv_path, img_dir):
       
        self.df = pd.read_csv(csv_path, sep=r'\|', engine='python', 
                              header=None, names=['image', 'index', 'caption'])
        self.img_dir = img_dir
        self.tokenizer = AutoTokenizer.from_pretrained(LocalConfig.TEXT_MODEL)
        self.transform = transforms.Compose([
            transforms.Resize((LocalConfig.IMG_SIZE, LocalConfig.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
       
        img_path = f"{self.img_dir}/{row.image.strip()}"
        img = Image.open(img_path).convert("RGB")
        # The tokenizer returns a BatchEncoding.
        text: BatchEncoding = self.tokenizer(
            row.caption,
            max_length=LocalConfig.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "images": self.transform(img),
            "input_ids": text.data["input_ids"].squeeze(0),
            "attention_mask": text.data["attention_mask"].squeeze(0)
        }

def get_dataloader(csv_path, img_dir, batch_size=32):
    dataset = Flickr30kDataset(csv_path, img_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
