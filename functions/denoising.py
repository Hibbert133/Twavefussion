import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)

            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds



# def ddim_steps(x, seq, model, transformer, betas, eta=0.0):
#     with torch.no_grad():
#         n = x.size(0)
#         #print(x.shape)
#         seq_next = [-1] + list(seq[:-1])
#         xs = [x]
#         x0_preds = []
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(betas, t.long())
#             atm1 = compute_alpha(betas, next_t.long())
#             beta_t = 1 - at / atm1
#             x = xs[-1].to('cuda')
#             #print(x.shape)

#             # 使用模型预测噪声
#             output = model(x, t.float())
#             e = output
            
#             # 使用 Transformer 生成动态噪声
            
#             custom_noise = generate_custom_noise_with_transformer(x, transformer)
            
#             # 使用 DDIM 采样步骤
#             x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
#             x0_from_e = torch.clamp(x0_from_e, -1, 1)
#             x0_preds.append(x0_from_e.to('cpu'))

#             mean_eps = (
#                 (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
#             ) / (1.0 - at)

#             mean = mean_eps
#             noise = custom_noise  # 使用 Transformer 生成的自定义噪声
#             mask = 1 - (t == 0).float()
#             # mask = mask.view(-1, 1, 1, 1)
#             mask = mask.view(-1, 1, 1)  # 适配 x 的三维形状

#             logvar = beta_t.log()
#             sample = mean + mask * torch.exp(0.5 * logvar) * noise
#             xs.append(sample.to('cpu'))
#     return xs, x0_preds

def generate_custom_noise_with_transformer(x, transformer):
    mean_x, scale_factor_x = transformer(x)
    noise = torch.randn_like(x)
    custom_noise = noise * scale_factor_x + mean_x
    return custom_noise

def ddim_steps(x, seq, model, betas, eta=0.0):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            # 使用模型预测噪声
            output = model(x, t.float())
            e = output

            # 使用 Transformer 生成动态噪声
            #custom_noise = generate_custom_noise_with_transformer(x, transformer)

            # 使用 ODE Solver 改进的 DDIM 采样步骤
            # 使用 OSE（ODE Solving Efficiency）方法，例如Euler或RK4求解器
            # Predict step: 使用估算的噪声和模型输出
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))

            # Mean prediction step - 用于 ODE 的估计
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            #noise = custom_noise  # 使用 Transformer 生成的自定义噪声
            noise = torch.randn_like(x)
            # noise = torch.randn(x.shape[0], x.shape[1], x.shape[2] * 2).to(x.device)
            # dwt1d = DWT1DForward(J=1, mode='zero', wave='db1').to(x.device)
            # coeffs_noise = dwt1d(noise)
            # _, cd__list_noise = coeffs_noise
            # noise = cd__list_noise[0]

    
            # Correction step: 根据 ODE 方案进行校正
            if i > 0:
                # 这里可以插入 ODE 校正策略，例如通过 RK4 方法进行微小调整
                mean = mean + (1 - beta_t).sqrt() * noise  # 简化的欧拉步

            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1)  # 适配 x 的三维形状

            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))

    return xs, x0_preds
