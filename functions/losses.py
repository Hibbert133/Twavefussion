import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)

    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()


    output = model(x, t.float()) # Predicted noise
    #print(output.shape)


    if keepdim:

        return x,(e - output).square().sum(dim=(0, 1, 2))
    else:

        return x,(e - output).square().sum(dim=(0, 1, 2)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
