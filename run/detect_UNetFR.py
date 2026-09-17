"""Run inference with the bundled frequency-regularized UNet checkpoint."""

import os
import tarfile
from pathlib import Path

import torch

if __package__:
    from .dice_score import multiclass_dice_coeff
    from .unet import UNet
else:
    from dice_score import multiclass_dice_coeff
    from unet import UNet


def loadFiles_plus(path_im, keyword=""):
    """Return matching file names and their full paths in sorted order."""

    path_im = os.fspath(path_im)
    files = sorted(os.listdir(path_im))

    file_names = []
    full_paths = []
    for file_name in files:
        if keyword in file_name:
            file_names.append(file_name)
            full_paths.append(path_im + "/" + file_name)

    return file_names, full_paths


def model2File(model, save_path):
    """Save a model state dictionary using sparse tensors."""

    sparse_state = {
        name: value.detach().to_sparse() for name, value in model.state_dict().items()
    }
    torch.save(sparse_state, save_path)


def file2Model(model, save_path):
    """Load a sparse state dictionary into ``model`` and return the model."""

    sparse_state = torch.load(save_path)
    dense_state = {name: value.to_dense() for name, value in sparse_state.items()}
    model.load_state_dict(dense_state)
    return model


def countParams(model):
    """Count non-zero weights and biases in a model."""

    total_weights = 0
    total_biases = 0

    for layer in model.modules():
        weight = getattr(layer, "weight", None)
        if weight is not None:
            total_weights += torch.sum(weight.data.abs() > 0).item()

        bias = getattr(layer, "bias", None)
        if bias is not None:
            total_biases += torch.sum(bias.data.abs() > 0).item()

    return total_weights, total_biases


def cleanUNet(model):
    """Remove pruning metadata after applying its masks to the weights."""

    for layer in model.modules():
        drop_mask = getattr(layer, "IDROP", None)
        weight = getattr(layer, "weight", None)
        if drop_mask is not None and weight is not None:
            idx = drop_mask.abs() < 1e-22
            weight.data[idx] = 0

        if hasattr(layer, "IDROP"):
            layer.IDROP = None
        if hasattr(layer, "ZMAT"):
            layer.ZMAT = None
        if hasattr(layer, "IMAT"):
            layer.IMAT = None


def extract_checkpoint(archive_path, output_dir):
    """Extract the checkpoint archive into the demo directory."""

    with tarfile.open(archive_path, mode="r:xz") as archive:
        archive.extractall(output_dir)


def run_demo(project_dir=None):
    """Run the bundled segmentation demo and display each prediction."""

    import imageio
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    if project_dir is None:
        project_dir = Path(__file__).resolve().parent
    else:
        project_dir = Path(project_dir).resolve()

    archive_path = project_dir / "unet_fr.tar.xz"
    checkpoint_path = project_dir / "unet_fr.pt"
    extract_checkpoint(archive_path, project_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    cleanUNet(net)
    net = file2Model(net, str(checkpoint_path))
    net.to(device=device)

    num_weights, num_biases = countParams(net)
    print("number of nonzero parameters in weight:", num_weights)
    print("number of nonzero parameters in bias  :", num_biases)

    _, image_paths = loadFiles_plus(project_dir / "testimgs" / "input", "png")
    _, ground_truth_paths = loadFiles_plus(project_dir / "testimgs" / "mask", "png")

    dice_score = 0
    plt.figure(figsize=(12, 4))

    for index in range(len(image_paths)):
        image_path = image_paths[index]
        ground_truth_path = ground_truth_paths[index]
        image = (
            torch.tensor(
                imageio.imread(image_path),
                dtype=torch.float32,
            )
            / 255.0
        )
        ground_truth = (
            torch.tensor(
                imageio.imread(ground_truth_path),
                dtype=torch.float32,
            )
            / 255.0
        )

        print("segmenting files:", image_path)

        image = (
            image.permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
            )
        )
        ground_truth = ground_truth.unsqueeze(0).to(
            device=device,
            dtype=torch.long,
        )

        mask_prediction = net(image)
        displayed_prediction = mask_prediction.argmax(dim=1).squeeze()
        displayed_ground_truth = ground_truth.squeeze()

        ground_truth_one_hot = (
            F.one_hot(
                ground_truth,
                net.n_classes,
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        prediction_one_hot = (
            F.one_hot(
                mask_prediction.argmax(dim=1),
                net.n_classes,
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        dice_score += multiclass_dice_coeff(
            prediction_one_hot[:, 1:],
            ground_truth_one_hot[:, 1:],
            reduce_batch_first=False,
        )

        plt.subplot(1, 3, 1)
        plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.title("img")

        plt.subplot(1, 3, 2)
        plt.imshow(displayed_ground_truth.detach().cpu().numpy())
        plt.title("Groundtruth")

        plt.subplot(1, 3, 3)
        plt.imshow(displayed_prediction.detach().cpu().numpy())
        plt.title("Binary Mask")

        plt.pause(0.1)

    print("average Dice Score:", dice_score.item() / len(image_paths))


def main():
    """CLI entry point for the demo."""

    run_demo()


if __name__ == "__main__":
    main()
