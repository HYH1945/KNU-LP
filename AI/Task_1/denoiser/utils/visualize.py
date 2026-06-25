import numpy as np
import matplotlib.pyplot as plt
import torch

def show_samples_for_method(method_name: str, sample_results: list, sample_count: int = 3):
    sample_count = min(sample_count, len(sample_results))
    selected = sample_results[:sample_count]

    fig, axes = plt.subplots(sample_count, 3, figsize=(9, 3 * sample_count))

    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, sample in enumerate(selected):
        axes[i, 0].imshow(sample["input"], cmap="gray")
        axes[i, 0].set_title(f"Noisy Input\n{sample['filename']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sample["output"], cmap="gray")
        axes[i, 1].set_title(f"{method_name} Output")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(sample["target"], cmap="gray")
        axes[i, 2].set_title("Target (GT)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def show_comparison_across_methods(method_samples_dict: dict, sample_index: int = 0):
    # 한 샘플에 대해 여러 방법을 한 번에 비교
    methods = list(method_samples_dict.keys())
    num_methods = len(methods)

    fig, axes = plt.subplots(1, num_methods + 2, figsize=(3 * (num_methods + 2), 3))

    # 기준 샘플은 첫 method에서 가져옴
    first_method = methods[0]
    sample = method_samples_dict[first_method][sample_index]

    axes[0].imshow(sample["input"], cmap="gray")
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")

    for i, method in enumerate(methods, start=1):
        sample_m = method_samples_dict[method][sample_index]
        axes[i].imshow(sample_m["output"], cmap="gray")
        axes[i].set_title(method)
        axes[i].axis("off")

    axes[-1].imshow(sample["target"], cmap="gray")
    axes[-1].set_title("Target")
    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_samples_torch(model: torch.nn.Module, dataloader, device: str, num_samples: int = 3, title="Model Output"):
    model.eval()

    batch = next(iter(dataloader))
    x = batch["input"].to(device)
    y = batch["target"].to(device)
    filenames = batch["filename"]

    pred = model(x)

    num_samples = min(num_samples, x.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        inp_img = x[i].cpu().squeeze(0).numpy()
        pred_img = pred[i].cpu().squeeze(0).numpy()
        target_img = y[i].cpu().squeeze(0).numpy()

        axes[i, 0].imshow(inp_img, cmap="gray")
        axes[i, 0].set_title(f"Noisy Input\n{filenames[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_img, cmap="gray")
        axes[i, 1].set_title(title)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target_img, cmap="gray")
        axes[i, 2].set_title("Target (GT)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
