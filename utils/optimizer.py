import torch


def build_optimizer(config, model, logger):
    opt_lower = config.type.lower()
    optimizer = None
    optim_params = set_weight_decay(model, logger)
    if opt_lower == 'adam':
        optimizer = torch.optim.Adam(
            optim_params,
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def set_weight_decay(model, logger):
    optim_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            optim_params.append(param)
        else:
            logger.info('Params [{:s}] will not optimize.'.format(name))
    return optim_params
