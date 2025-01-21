from yacs.config import CfgNode


def get_config(args):
    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    if not _check_args('cfg'):
        raise ValueError(f'`cfg` must be specified')

    f = open(args.cfg)
    cfg = CfgNode().load_cfg(f)
    cfg.freeze()
    return cfg
