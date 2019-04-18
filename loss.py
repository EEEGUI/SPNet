import torch
import torch.nn.functional as F
import pytorch_ssim
from utils import get_interp_size

def sim_depth_loss(y_true, y_pred):
    c = 0.2 * torch.max(torch.abs(y_pred-y_true))
    loss = torch.abs(y_true-y_pred) * (torch.abs(y_pred - y_true) <= c).float() + (torch.pow(y_true-y_pred, 2) + c**2)/(2*c) * (torch.abs(y_pred-y_true) > c).float()

    loss = torch.mean(loss)

    return loss

def depth_loss(y_true, y_pred, theta=0.1, max_depth_val=1000.0 / 10.0):
    if y_true.size()[-1] != y_pred.size()[-1] or y_true.size()[-2] != y_pred.size()[-2]:
        y_pred = F.interpolate(
            y_pred,
            size=get_interp_size(y_pred, z_factor=4),
            mode="bilinear",
            align_corners=True,
        )
    l_depth = torch.mean(torch.abs(y_true-y_pred))

    dx_pred, dy_pred = _gradient(y_pred)
    dx_true, dy_true = _gradient(y_true)

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_true - dx_pred))

    ssim_loss = pytorch_ssim.SSIM(device=torch.device('cuda'))
    l_ssim = torch.clamp(1 - ssim_loss(y_true, y_pred), 0, 1)

    return theta * l_depth + l_edges + l_ssim


def _gradient(x):
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    return torch.abs(r-l), torch.abs(t-b)


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)



if __name__ == '__main__':
    yt = (torch.randn(4, 1, 348, 1280))
    yp = (torch.randn(4, 1, 348, 1280))
    print(depth_loss(yp, yt))
