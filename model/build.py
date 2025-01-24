from .nets.SwinIR import SwinIR


def build_model(config):
    model_type = config.type
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
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    return model

