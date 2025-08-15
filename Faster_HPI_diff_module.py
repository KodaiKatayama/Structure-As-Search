import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
Warning: Results produced on NVIDIA V100 GPUs may differ from those on newer A100 or H100 GPUs, even when running the same code with the same inputs. This is due to architectural and default precision differences—A100 and H100 introduce TensorFloat-32 (TF32) for FP32 matrix multiplications, support bfloat16, and often use different fused-kernel implementations than V100. These changes can alter rounding, accumulation order, and numerical precision, which can lead to small but noticeable output differences. If strict reproducibility across GPU types is required, you should disable TF32, fix precision modes, and enforce deterministic algorithms.
'''

# Optional: enable faster matmuls on NVIDIA (keeps accuracy for GNN-ish ops)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def GCN_diffusion(W, order, feature, device):
    """
    Perform GCN diffusion (GPU-optimized, same API).

    Args:
        W: [B, N, N]
        order: int
        feature: [B, N, N]
        device: torch.device
    Returns:
        tuple(Tensors): length=order, each [B, N, N]
    """
    W = W.to(device, non_blocking=True)
    feature = feature.to(device, non_blocking=True)

    B, N, _ = W.shape
    use_amp = (device.type == "cuda")

    # New autocast API
    with torch.amp.autocast("cuda", enabled=use_amp):
        A_gcn = W.clone()
        diag_idx = torch.arange(N, device=device)
        A_gcn[:, diag_idx, diag_idx] += 1

        degrees = A_gcn.sum(dim=2, keepdim=True)
        D_inv_sqrt = degrees.clamp_min(1e-12).rsqrt()

        A_norm = D_inv_sqrt * A_gcn * D_inv_sqrt.transpose(-1, -2)

        feat = feature
        out = []
        A_norm = A_norm.contiguous()
        for _ in range(order):
            feat = torch.bmm(A_norm, feat.contiguous())
            out.append(feat)

    return tuple(t.contiguous() for t in out)


def SCT1stv2(W, order, feature, device):
    """
    Perform SCT diffusion (GPU-optimized, same API).

    Args:
        W: [B, N, N]
        order: int
        feature: [B, N, N]
        device: torch.device
    Returns:
        tuple(Tensors): length=order, each [B, N, N] (band-pass like)
    """
    W = W.to(device, non_blocking=True)
    feature = feature.to(device, non_blocking=True)

    B, N, _ = W.shape
    use_amp = (device.type == "cuda")

    # New autocast API
    with torch.amp.autocast("cuda", enabled=use_amp):
        # D^{-1}
        deg = W.sum(dim=2, keepdim=True)
        D_inv = deg.add_(1e-6).reciprocal_()

        # Precompute W * D^{-1}
        W_Dinv = W * D_inv.transpose(-1, -2)

        iteration = 1 << order  # 2**order
        scale_marks = [(1 << i) - 1 for i in range(order + 1)]

        feat = feature
        sct_diff = [None] * (order + 1)

        W_Dinv = W_Dinv.contiguous()
        for i in range(iteration):
            nxt = torch.bmm(W_Dinv, feat.contiguous())

            # --- dtype-safe blend to avoid RuntimeError in mixed precision ---
            if nxt.dtype != feat.dtype:
                feat = feat.to(nxt.dtype)
            # lerp(x,y,0.5) == (x + y)/2; use add for fewer surprises
            feat = torch.add(feat, nxt).mul_(0.5)
            # ----------------------------------------------------------------

            if i in scale_marks:
                sct_diff[scale_marks.index(i)] = feat

        out = []
        for i in range(len(scale_marks) - 1):
            out.append((sct_diff[i] - sct_diff[i + 1]).contiguous())
            sct_diff[i] = None

    return tuple(out)


def scattering_diffusion(sptensor, feature, sctorder=7):
    """
    sptensor: [batchsize, n, n]
    feature: [batchsize, n, x]
    """
    # assumes `device` is defined in the calling scope as before
    return SCT1stv2(sptensor, sctorder, feature, device)
