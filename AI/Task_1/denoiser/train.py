import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dncnn import DnCNN
from dataset.plate_dataset import MixedPlateDenoiseDataset, MixedInputDataset, PlateDenoiseDataset

def calculate_psnr(img1, img2):
    """
    img1, img2: torch.Tensor, range [0, 1]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def train(args):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 설정
    print(f"Loading datasets from: {args.data_dir}")
    train_dataset = MixedPlateDenoiseDataset(
        split_root=os.path.join(args.data_dir, "train"),
        image_size=tuple(args.image_size),
        identity_ratio=args.identity_ratio,
        channels=args.channels
    )
    val_base = PlateDenoiseDataset(
        split_root=os.path.join(args.data_dir, "val"),
        image_size=tuple(args.image_size),
        channels=args.channels
    )
    val_dataset = MixedInputDataset(
        base_dataset=val_base,
        clean_ratio=args.identity_ratio,
        seed=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train dataset size: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val dataset size: {len(val_dataset)} ({len(val_loader)} batches)")

    # 모델 초기화
    model = DnCNN(in_channels=args.channels, depth=args.depth, features=args.features)
    model.to(device)

    # 손실함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_psnr = 0.0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # PSNR 계산
                for i in range(inputs.size(0)):
                    val_psnr += calculate_psnr(outputs[i], targets[i]).item()

        val_loss = val_loss / len(val_dataset)
        val_psnr = val_psnr / len(val_dataset)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | Time: {epoch_time:.2f}s")

        # 최적 모델 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_psnr = val_psnr
            best_model_path = os.path.join(args.checkpoint_dir, "best_color_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Best model saved! (Val Loss: {best_val_loss:.6f}, PSNR: {best_psnr:.2f} dB)")

        # 주기적인 저장
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Training finished. Best Val Loss: {best_val_loss:.6f} | Best Val PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DnCNN Color/Grayscale Denoising Training Script")
    parser.add_argument("--data_dir", type=str, default="data/dataset_color", help="path to dataset root")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels (1 for grayscale, 3 for RGB)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[48, 128], help="image resolution [height, width]")
    parser.add_argument("--depth", type=int, default=17, help="depth of DnCNN")
    parser.add_argument("--features", type=int, default=64, help="features channel of DnCNN layers")
    parser.add_argument("--identity_ratio", type=float, default=0.4, help="ratio of low-noise input to pass identity mapping")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers for DataLoader")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=10, help="frequency of epochs to save general checkpoints")
    parser.add_argument("--cpu", action="store_true", help="force to use CPU instead of GPU")

    args = parser.parse_args()
    train(args)
