import torch


def box_iou(
    boxes1,
    boxes2,
    eps=1e-7
):
    """
    Calculate IoU between corresponding boxes.

    boxes format:
        [x1, y1, x2, y2]

    Args:
        boxes1: [N, 4]
        boxes2: [N, 4]

    Returns:
        IoU: [N]
    """

    x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0]
    )

    y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1]
    )

    x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2]
    )

    y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3]
    )

    intersection_width = (
        x2 - x1
    ).clamp(
        min=0
    )

    intersection_height = (
        y2 - y1
    ).clamp(
        min=0
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0])
        .clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1])
        .clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0])
        .clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1])
        .clamp(min=0)
    )

    union = (
        area1
        + area2
        - intersection
    )

    iou = intersection / (
        union + eps
    )

    return iou


def iou_loss(
    pred_boxes,
    target_boxes
):
    """
    IoU loss.

    Loss = 1 - IoU
    """

    if pred_boxes.numel() == 0:

        return pred_boxes.sum() * 0.0

    iou = box_iou(
        pred_boxes,
        target_boxes
    )

    return (
        1.0 - iou
    ).mean()