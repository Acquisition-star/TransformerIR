from yacs.config import CfgNode


def _check_args(args, name):
    if hasattr(args, name) and eval(f'args.{name}'):
        return True
    return False


def update_config(args, config):
    config.defrost()
    if hasattr(args, 'dist'):
        config.dist = args.dist
    config.freeze()


def get_config(args):
    if not _check_args(args, 'cfg'):
        raise ValueError(f'`cfg` must be specified')

    # 显式指定编码为 utf-8
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = CfgNode().load_cfg(f)
    cfg.freeze()
    update_config(args, cfg)
    return cfg
