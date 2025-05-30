from .nets.Stripformer import Stripformer
from .nets.SwinIR import SwinIR
from .nets.Uformer import Uformer
from .nets.NAFNet import NAFNet
from .nets.MB_TaylorFormerV2 import MB_TaylorFormer
from .nets.Restormer import Restormer
from .TransformerIR import TransformerIR
from .restormer_baseline import Restormer_Baseline
from .restormer_conv import Restormer_Baseline as Restormer_Conv


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
    elif 'mb_taylorformer_v2' in model_type:
        model = MB_TaylorFormer(
            inp_channels=config.in_chans,
            dim=config.embed_dim,
            num_blocks=config.num_blocks,
            num_refinement_blocks=config.num_refinement_blocks,
            heads=config.heads,
            ffn_expansion_factor=config.ffn_expansion_factor,
            bias=config.bias,
            LayerNorm_type=config.LayerNorm_type,
            dual_pixel_task=config.dual_pixel_task,
            num_path=config.num_path,
            qk_norm=config.qk_norm,
            offset_clamp=config.offset_clamp,
        )
    elif model_type == 'restormer':
        model = Restormer(
            inp_channels=config.in_chans,
            out_channels=config.out_chans,
            dim=config.embed_dim,
            num_blocks=config.num_blocks,
            num_refinement_blocks=config.num_refinement_blocks,
            heads=config.heads,
            ffn_expansion_factor=config.ffn_expansion_factor,
            bias=config.bias,
            LayerNorm_type=config.LayerNorm_type,
            dual_pixel_task=config.dual_pixel_task,
        )
    elif model_type == 'baseline':
        model = TransformerIR(
            dim=config.dim,
            embedding_dim=config.embedding_dim,
            bias=config.bias,
            middle_blks=config.middle_blks,
            encoder_blk_nums=config.encoder_blk_nums,
            decoder_blk_nums=config.decoder_blk_nums,
            attn_type=config.attn_type,
            attn_config=config.attn_config,
        )
    elif model_type == 'restormer_baseline':
        model = Restormer_Baseline(
            inp_channels=config.in_chans,
            dim=config.dim,
            bias=config.bias,
            num_blocks=config.num_blocks,
            num_refinement_blocks=config.num_refinement_blocks,
            ffn_expansion_factor=config.ffn_expansion_factor,
            LayerNorm_type=config.LayerNorm_type,
            attn_type=config.attn_type,
            attn_config=config.attn_config,
        )
    elif model_type == 'restormer_conv':
        model = Restormer_Conv(
            inp_channels=config.in_chans,
            dim=config.dim,
            bias=config.bias,
            num_blocks=config.num_blocks,
            num_refinement_blocks=config.num_refinement_blocks,
            ffn_expansion_factor=config.ffn_expansion_factor,
            LayerNorm_type=config.LayerNorm_type,
            attn_type=config.attn_type,
            attn_config=config.attn_config,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    return model
