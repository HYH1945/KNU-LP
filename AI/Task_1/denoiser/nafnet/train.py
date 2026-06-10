import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from model import NAFNet

try:
    import kornia
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class SimplePlateDataset(Dataset):
    def __init__(self, root_dir, image_size=(48, 128), transform=None, identity_ratio=0.0):
        self.clean_dir = os.path.join(root_dir, "clean")
        self.noisy_dir = os.path.join(root_dir, "noisy")
        self.image_size = image_size
        self.transform = transform
        self.identity_ratio = identity_ratio
        
        # files
        self.files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(('.png', '.jpg'))])
        
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean_path = os.path.join(self.clean_dir, fname)
        noisy_path = os.path.join(self.noisy_dir, fname)
        
        clean_img = Image.open(clean_path).convert('RGB')
        
        import random
        if random.random() < self.identity_ratio:
            noisy_img = clean_img.copy()
        elif os.path.exists(noisy_path):
            noisy_img = Image.open(noisy_path).convert('RGB')
        else:
            noisy_img = clean_img.copy()

        clean_tensor = self.to_tensor(clean_img)
        noisy_tensor = self.to_tensor(noisy_img)

        return {'input': noisy_tensor, 'target': clean_tensor}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    print(f"Loading datasets from: {args.data_dir}")
    train_dataset = SimplePlateDataset(
        root_dir=os.path.join(args.data_dir, "train"),
        image_size=tuple(args.image_size),
        identity_ratio=args.identity_ratio
    )
    val_dataset = SimplePlateDataset(
        root_dir=os.path.join(args.data_dir, "val"),
        image_size=tuple(args.image_size),
        identity_ratio=0.0 # val 셋은 온전히 평가를 위해 0% 적용
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    print(f"Train dataset size: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val dataset size: {len(val_dataset)} ({len(val_loader)} batches)")

    model = NAFNet(in_channels=args.channels, out_channels=args.channels, width=args.width, 
                   enc_blk_nums=args.enc_blks, middle_blk_num=args.mid_blk, dec_blk_nums=args.dec_blks)
    model.to(device)

    criterion_charbonnier = CharbonnierLoss(eps=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_psnr = 0.0

    print("Starting experimental NAFNet training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_c = criterion_charbonnier(outputs, targets)
            loss = loss_c

            if HAS_KORNIA and args.use_ssim_loss:
                loss_ssim = kornia.losses.ssim_loss(outputs, targets, window_size=11, reduction='mean')
                loss = loss + 0.1 * loss_ssim

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_dataset)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)

                outputs = model(inputs)
                loss = criterion_charbonnier(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                for i in range(inputs.size(0)):
                    val_psnr += calculate_psnr(outputs[i].clamp(0,1), targets[i].clamp(0,1)).item()

        val_loss = val_loss / len(val_dataset)
        val_psnr = val_psnr / len(val_dataset)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss(Charb): {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.2f}s")

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            best_model_path = os.path.join(args.checkpoint_dir, f"best_nafnet_model_ratio{int(args.identity_ratio*100)}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Best NAFNet model saved! (PSNR: {best_psnr:.2f} dB)")

    print(f"Training finished. Best Val PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAFNet Denoising Experimental Training Script")
    parser.add_argument("--data_dir", type=str, default="../../dataset_denoising_color", help="path to dataset root")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--image_size", type=int, nargs=2, default=[48, 128], help="image resolution [height, width]")
    parser.add_argument("--width", type=int, default=32, help="width (channels) of NAFNet")
    parser.add_argument("--enc_blks", type=int, nargs='+', default=[1, 1, 1, 14], help="encoder block nums")
    parser.add_argument("--mid_blk", type=int, default=1, help="middle block nums")
    parser.add_argument("--dec_blks", type=int, nargs='+', default=[1, 1, 1, 1], help="decoder block nums")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--identity_ratio", type=float, default=0.0, help="ratio of clean data to pass identity mapping")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers for DataLoader")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_nafnet", help="directory to save NAFNet checkpoints")
    parser.add_argument("--use_ssim_loss", action="store_true", help="use SSIM loss via kornia")
    parser.add_argument("--cpu", action="store_true", help="force to use CPU instead of GPU")

    args = parser.parse_args()
    train(args)
