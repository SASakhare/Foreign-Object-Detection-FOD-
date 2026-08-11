import torch
import torch.nn.functional as F


def make_grid_points(
    height,
    width,
    stride,
    device
):
    """
    Create center points for a feature map.

    Returns:
        [H*W, 2]

    Coordinates are in image pixels.
    """

    y, x = torch.meshgrid(
        torch.arange(
            height,
            device=device
        ),
        torch.arange(
            width,
            device=device
        ),
        indexing="ij"
    )

    points = torch.stack(
        [
            (x + 0.5) * stride,
            (y + 0.5) * stride
        ],
        dim=-1
    )

    return points.reshape(
        -1,
        2
    )


def decode_distribution(
    pred,
    reg_max
):
    """
    Decode DFL regression predictions.

    Args:
        pred:
            [B, 4 * reg_max, H, W]

    Returns:
        distances:
            [B, H*W, 4]
    """

    batch_size, _, height, width = (
        pred.shape
    )

    pred = pred.reshape(
        batch_size,
        4,
        reg_max,
        height,
        width
    )

    pred = pred.permute(
        0,
        3,
        4,
        1,
        2
    )

    pred = pred.reshape(
        batch_size,
        height * width,
        4,
        reg_max
    )

    probabilities = F.softmax(
        pred,
        dim=-1
    )

    bins = torch.arange(
        reg_max,
        device=pred.device,
        dtype=pred.dtype
    )

    distances = (
        probabilities
        * bins
    ).sum(
        dim=-1
    )

    return distances


def distances_to_boxes(
    points,
    distances
):
    """
    Convert LTRB distances to XYXY boxes.

    Args:
        points:
            [N, 2]

        distances:
            [B, N, 4]

    Returns:
        boxes:
            [B, N, 4]
    """

    x = points[:, 0]
    y = points[:, 1]

    left = distances[..., 0]
    top = distances[..., 1]
    right = distances[..., 2]
    bottom = distances[..., 3]

    x1 = (
        x[None, :]
        - left
    )

    y1 = (
        y[None, :]
        - top
    )

    x2 = (
        x[None, :]
        + right
    )

    y2 = (
        y[None, :]
        + bottom
    )

    boxes = torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=-1
    )

    return boxes




def assign_targets(
    points,
    gt_boxes,
    gt_labels,
    image_width,
    image_height
):
    """
    Simple center-based target assignment.

    Args:
        points:
            [N, 2] image-space points

        gt_boxes:
            [M, 4] XYXY

        gt_labels:
            [M]

    Returns:
        target_boxes:
            [N, 4]

        target_labels:
            [N]

        foreground:
            [N] bool
    """

    num_points = points.shape[0]

    device = points.device

    target_boxes = torch.zeros(
        num_points,
        4,
        device=device
    )

    target_labels = torch.zeros(
        num_points,
        dtype=torch.long,
        device=device
    )

    foreground = torch.zeros(
        num_points,
        dtype=torch.bool,
        device=device
    )

    if gt_boxes.numel() == 0:

        return (
            target_boxes,
            target_labels,
            foreground
        )

    for box, label in zip(
        gt_boxes,
        gt_labels
    ):

        x1, y1, x2, y2 = box

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        inside = (
            (points[:, 0] >= x1)
            &
            (points[:, 0] <= x2)
            &
            (points[:, 1] >= y1)
            &
            (points[:, 1] <= y2)
        )

        candidate_indices = (
            torch.where(inside)[0]
        )

        if candidate_indices.numel() == 0:

            distances = (
                points[:, 0]
                - center_x
            ) ** 2 + (
                points[:, 1]
                - center_y
            ) ** 2

            index = torch.argmin(
                distances
            )

        else:

            distances = (
                points[
                    candidate_indices,
                    0
                ]
                - center_x
            ) ** 2 + (
                points[
                    candidate_indices,
                    1
                ]
                - center_y
            ) ** 2

            index = (
                candidate_indices[
                    torch.argmin(
                        distances
                    )
                ]
            )

        target_boxes[index] = box

        target_labels[index] = label

        foreground[index] = True

    return (
        target_boxes,
        target_labels,
        foreground
    )