from yacs.config import CfgNode


def _check_args(args, name):
    if hasattr(args, name) and eval(f'args.{name}'):
        return True
    return False


def replace_path(path):
    if path == 'None':
        return 'None'
    # 将路径中的反斜杠替换为正斜杠
    path = path.replace('\\', '/')
    # 找到 'SIDD/train/groundtruth' 的位置
    index = path.find('SIDD')
    if index != -1:
        # 替换为新的路径前缀
        new_path = '/root/autodl-tmp/TrainData/' + path[index:]
        return new_path
    return path


def update_config(args, config):
    config.defrost()
    if hasattr(args, 'dist'):
        config.dist = args.dist
    if hasattr(args, 'dataloader_workers'):
        config.datasets.dataloader_num_workers = args.dataloader_workers
    if hasattr(args, 'batch_size'):
        config.datasets.dataloader_batch_size = args.batch_size
    if hasattr(args, 'epochs'):
        config.train.num_epochs = args.epochs
        if hasattr(config.scheduler, 'T_max'):
            config.scheduler.T_max = args.epochs
    if hasattr(args, 'autodl'):
        if args.autodl:
            config.datasets.train.H_path = [replace_path(p) for p in config.datasets.train.H_path]
            config.datasets.train.L_path = [replace_path(p) for p in config.datasets.train.L_path]
            config.datasets.val.H_path = [replace_path(p) for p in config.datasets.val.H_path]
            config.datasets.val.L_path = [replace_path(p) for p in config.datasets.val.L_path]
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
