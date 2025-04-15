from collections import OrderedDict

from .nets.Stripformer import Stripformer
from .nets.SwinIR import SwinIR
from .nets.Uformer import Uformer
from .nets.NAFNet import NAFNet
from .restormer_baseline import Restormer_Baseline

from utils.checkpoint import *


class BaseModelLoader:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        pass

    def load_cpk_model(self, model_pth, logger=None):
        return load_checkpoint_model(self.build_model(), model_pth, logger)

    def load_model(self, model_pth):
        pass


class SwinIR_loader(BaseModelLoader):
    def build_model(self):
        model = SwinIR(
            img_size=self.config.img_size,
            in_chans=self.config.in_chans,
            embed_dim=self.config.embed_dim,
            depths=self.config.depths,
            num_heads=self.config.num_heads,
            window_size=self.config.window_size,
            mlp_ratio=self.config.mlp_ratio,
            upscale=self.config.upscale,
            img_range=self.config.img_range,
            upsampler=self.config.upsampler,
            resi_connection=self.config.resi_connection,
        )
        return model

    def load_model(self, model_pth):
        model = self.build_model()
        param_key_g = 'params'
        pretrained_model = torch.load(model_pth)
        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)


class Uformer_loader(BaseModelLoader):
    def build_model(self):
        model = Uformer(
            img_size=self.config.img_size,
            embed_dim=self.config.embed_dim,
            win_size=self.config.window_size,
            token_projection=self.config.token_projection,
            token_mlp=self.config.token_mlp,
            depths=self.config.depths,
            modulator=self.config.modulator,
        )
        return model

    def load_model(self, model_pth):
        model = self.build_model()
        cpk = torch.load(model_pth)
        try:
            model.load_state_dict(cpk["state_dict"])
        except:
            state_dict = cpk["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        return model


class NAFNet_loader(BaseModelLoader):
    def build_model(self):
        model = NAFNet(
            width=self.config.img_size,
            middle_blk_num=self.config.middle_blk_num,
            enc_blk_nums=self.config.enc_blk_nums,
            dec_blk_nums=self.config.dec_blk_nums,
        )
        return model

    def load_model(self, model_pth):
        model = self.build_model()
        load_net = torch.load(model_pth, map_location=lambda storage, loc: storage)
        load_net = load_net['params']
        model.load_state_dict(load_net, strict=True)


class Stripformer_loader(BaseModelLoader):
    def build_model(self):
        return Stripformer()

    def load_model(self, model_pth):
        model = self.build_model()
        model.load_state_dict(torch.load(model_pth))
        return model


class Restormer_Baseline_loader(BaseModelLoader):
    def build_model(self):
        model = Restormer_Baseline(
            inp_channels=self.config.in_chans,
            dim=self.config.dim,
            bias=self.config.bias,
            num_blocks=self.config.num_blocks,
            num_refinement_blocks=self.config.num_refinement_blocks,
            ffn_expansion_factor=self.config.ffn_expansion_factor,
            LayerNorm_type=self.config.LayerNorm_type,
            attn_type=self.config.attn_type,
            attn_config=self.config.attn_config,
        )
        return model


class ModelBuilder:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(loader_cls):
            cls._registry[name] = loader_cls
            return loader_cls

        return decorator

    @classmethod
    def get_model(cls, config):
        for key, loader_cls in cls._registry.items():
            if key == config.type:
                loader = loader_cls(config)
                return loader
        raise ValueError(f"No loader found for model {config.type}")


ModelBuilder.register('SwinIR')(SwinIR_loader)
ModelBuilder.register('Uformer')(Uformer_loader)
ModelBuilder.register('NAFNet')(NAFNet_loader)
ModelBuilder.register('Stripformer')(Stripformer_loader)
ModelBuilder.register('restormer_baseline')(Restormer_Baseline_loader)
