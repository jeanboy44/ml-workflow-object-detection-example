from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def load_image(img_path: Path) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def plot_detections(
    image, boxes, scores, labels, score_threshold=0.1, save_path: Path = None
):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        rect = patches.Rectangle(
            (x0, y0), w, h, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 5,
            f"{label}: {score:.2f}",
            bbox=dict(facecolor="yellow", alpha=0.5),
            fontsize=12,
        )
    plt.axis("off")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Detect result saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
