from collections import OrderedDict

from model.nets.Stripformer import Stripformer
from model.nets.SwinIR import SwinIR
from model.nets.Uformer import Uformer
from model.nets.NAFNet import NAFNet

from model.restormer_baseline import Restormer_Baseline

from utils.checkpoint import *

import argparse
from utils.config import get_config


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransformerIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='configs/Denoising/Baseline/Restormer_demo0.yaml',
                        help='path to config file')

    args = parser.parse_args()

    config = get_config(args)

    model = ModelBuilder.get_model(config.net).build_model()
    print(model)

#
# def build_model(config):
#     model_type = config.type.lower()
#     if model_type == 'swinir':
#         model = SwinIR(
#             img_size=config.img_size,
#             in_chans=config.in_chans,
#             embed_dim=config.embed_dim,
#             depths=config.depths,
#             num_heads=config.num_heads,
#             window_size=config.window_size,
#             mlp_ratio=config.mlp_ratio,
#             upscale=config.upscale,
#             img_range=config.img_range,
#             upsampler=config.upsampler,
#             resi_connection=config.resi_connection,
#         )
#     elif 'uformer' in model_type:
#         model = Uformer(
#             img_size=config.img_size,
#             embed_dim=config.embed_dim,
#             win_size=config.window_size,
#             token_projection=config.token_projection,
#             token_mlp=config.token_mlp,
#             depths=config.depths,
#             modulator=config.modulator,
#         )
#     elif 'nafnet' in model_type:
#         model = NAFNet(
#             width=config.img_size,
#             middle_blk_num=config.middle_blk_num,
#             enc_blk_nums=config.enc_blk_nums,
#             dec_blk_nums=config.dec_blk_nums,
#         )
#     elif 'stripformer' in model_type:
#         model = Stripformer()
#     elif 'mb_taylorformer_v2' in model_type:
#         model = MB_TaylorFormer(
#             inp_channels=config.in_chans,
#             dim=config.embed_dim,
#             num_blocks=config.num_blocks,
#             num_refinement_blocks=config.num_refinement_blocks,
#             heads=config.heads,
#             ffn_expansion_factor=config.ffn_expansion_factor,
#             bias=config.bias,
#             LayerNorm_type=config.LayerNorm_type,
#             dual_pixel_task=config.dual_pixel_task,
#             num_path=config.num_path,
#             qk_norm=config.qk_norm,
#             offset_clamp=config.offset_clamp,
#         )
#     elif model_type == 'restormer':
#         model = Restormer(
#             inp_channels=config.in_chans,
#             out_channels=config.out_chans,
#             dim=config.embed_dim,
#             num_blocks=config.num_blocks,
#             num_refinement_blocks=config.num_refinement_blocks,
#             heads=config.heads,
#             ffn_expansion_factor=config.ffn_expansion_factor,
#             bias=config.bias,
#             LayerNorm_type=config.LayerNorm_type,
#             dual_pixel_task=config.dual_pixel_task,
#         )
#     elif model_type == 'baseline':
#         model = TransformerIR(
#             dim=config.dim,
#             embedding_dim=config.embedding_dim,
#             bias=config.bias,
#             middle_blks=config.middle_blks,
#             encoder_blk_nums=config.encoder_blk_nums,
#             decoder_blk_nums=config.decoder_blk_nums,
#             attn_type=config.attn_type,
#             attn_config=config.attn_config,
#         )
#     elif model_type == 'restormer_baseline':
#         model = Restormer_Baseline(
#             inp_channels=config.in_chans,
#             dim=config.dim,
#             bias=config.bias,
#             num_blocks=config.num_blocks,
#             num_refinement_blocks=config.num_refinement_blocks,
#             ffn_expansion_factor=config.ffn_expansion_factor,
#             LayerNorm_type=config.LayerNorm_type,
#             attn_type=config.attn_type,
#             attn_config=config.attn_config,
#         )
#     else:
#         raise NotImplementedError(f"Unknown model: {model_type}")
#     return model
