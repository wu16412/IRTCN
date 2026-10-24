import torch
import torch.nn as nn

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        # 使用 in-place 操作避免创建临时张量，节省显存
        for name, param in module.named_parameters():
            if param.requires_grad:
                shadow = self.shadow[name]
                # resume 时 shadow 可能在 CPU，对齐到参数设备
                if shadow.device != param.device:
                    shadow = shadow.to(param.device, non_blocking=False)
                    self.shadow[name] = shadow
                # 等价于: shadow = mu * shadow + (1-mu) * param
                shadow.data.mul_(self.mu).add_(param.data, alpha=1. - self.mu)

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                shadow = self.shadow[name]
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                    self.shadow[name] = shadow
                param.data.copy_(shadow.data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict, device=None):
        """device: 若给定，将 shadow 挪到该设备（resume 从 CPU ckpt 加载时必须指定 GPU）。"""
        self.shadow = {}
        for name, val in state_dict.items():
            t = val.data if torch.is_tensor(val) and hasattr(val, "data") else val
            if torch.is_tensor(t):
                if device is not None:
                    t = t.to(device)
                else:
                    t = t.clone() if t.device.type == "cpu" else t
                self.shadow[name] = t
            else:
                self.shadow[name] = val
