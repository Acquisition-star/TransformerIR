from .nets.Stripformer import Stripformer
from .nets.SwinIR import SwinIR
from .nets.Uformer import Uformer
from .nets.NAFNet import NAFNet
from .TransformerIR import TransformerIR


def build_model(config):
    model_type = config.type.lower()
    if model_type == 'swinir':
        model = SwinIR(
            img_size=config.img_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            upscale=config.upscale,
            img_range=config.img_range,
            upsampler=config.upsampler,
            resi_connection=config.resi_connection,
        )
    elif 'uformer' in model_type:
        model = Uformer(
            img_size=config.img_size,
            embed_dim=config.embed_dim,
            win_size=config.window_size,
            token_projection=config.token_projection,
            token_mlp=config.token_mlp,
            depths=config.depths,
            modulator=config.modulator,
        )
    elif 'nafnet' in model_type:
        model = NAFNet(
            width=config.img_size,
            middle_blk_num=config.middle_blk_num,
            enc_blk_nums=config.enc_blk_nums,
            dec_blk_nums=config.dec_blk_nums,
        )
    elif 'stripformer' in model_type:
        model = Stripformer()
    elif model_type == 'baseline':
        model = TransformerIR(
            img_size=config.img_size,
            channels=config.channels,
            window_size=config.window_size,
            embedding_dim=config.embedding_dim,
            middle_blks=config.middle_blks,
            encoder_blk_nums=config.encoder_blk_nums,
            decoder_blk_nums=config.decoder_blk_nums,
            attn_type= config.attn_type if config.attn_type != 'None' else None,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    return model
