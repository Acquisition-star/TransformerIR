import torch

from .BaseAttentionLoader import BaseAttentionLoader


class AttentionBuilder:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(loader_cls):
            cls._registry[name] = loader_cls
            return loader_cls

        return decorator

    @classmethod
    def get_attention(cls, index, iter, dim, bias, attn_type, attn_config):
        for key, loader_cls in cls._registry.items():
            if key == attn_type:
                loader = loader_cls(index, iter, dim, bias, attn_config)
                return loader
        raise ValueError(f"No loader found for model {attn_type}")


class Identity_loader(BaseAttentionLoader):
    def load(self):
        return torch.nn.Identity()


AttentionBuilder.register('Identity')(Identity_loader)
