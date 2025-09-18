
import torch

def masked_l1_loss(pred, target):
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.abs(pred[mask] - target[mask]).mean()

def _ensure_2d(t: torch.Tensor, feat_dim: int) -> torch.Tensor:
    if t.dim() == 2: return t
    if t.dim() == 1: return t.view(-1, feat_dim)
    raise RuntimeError(f"Unsupported t.dim()={t.dim()}")

@torch.no_grad()
def compute_metrics(outputs, batch):
    res = {}
    # Classification
    logits = outputs['logits']
    y = batch.y
    pred = logits.argmax(dim=1)
    res['acc_sum'] = (pred == y).sum().item()
    res['n_cls']   = y.numel()

    # Estimation
    t = _ensure_2d(batch.t, 4)
    pose_tgt = t[:, 0:2]
    pose_pred = outputs['pose']
    mask_pose = torch.isfinite(pose_tgt)
    n_pose = mask_pose[:,0].sum().item()
    if n_pose > 0:
        mae = torch.abs(pose_pred - pose_tgt); mae[~mask_pose] = 0.0
        res['mae_pose2_sum'] = mae[:,0].sum().item()
        res['mae_pose6_sum'] = mae[:,1].sum().item()
    else:
        res['mae_pose2_sum'] = 0.0; res['mae_pose6_sum'] = 0.0
    res['n_pose'] = max(1, n_pose)

    # Depth
    depth_tgt  = t[:, 2:3]
    depth_pred = outputs['depth']
    mask_depth = torch.isfinite(depth_tgt)
    n_depth = mask_depth.sum().item()
    if n_depth > 0:
        res['mae_depth_sum'] = torch.abs(depth_pred - depth_tgt)[mask_depth].sum().item()
    else:
        res['mae_depth_sum'] = 0.0
    res['n_depth'] = max(1, n_depth)

    return res
