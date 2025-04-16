from torch.utils.data import DataLoader
from .dataset_sr import DatasetSR
from .dataset_denoising import Dataset_denoising_train, Dataset_denoising_val
from .dataset_deblur import Dataset_deblur_train, Dataset_deblur_val


def build_loader(config):
    dataset_type = config.datasets.dataset_type
    if dataset_type == 'sr':
        return build_loader_sr(config)
    elif dataset_type == 'denoising':
        return build_loader_denoising(config)
    elif dataset_type == 'deblur':
        return build_loader_deblur(config)
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))


def build_loader_sr(config):
    config.defrost()
    config.datasets.train.up_scale = config.scale
    config.datasets.train.n_channels = config.n_channels
    config.datasets.val.up_scale = config.scale
    config.datasets.val.n_channels = config.n_channels
    config.freeze()
    dataset_train, dataset_val = DatasetSR(config.datasets.train), DatasetSR(config.datasets.val, is_train=False)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.datasets.dataloader_batch_size,
        shuffle=config.datasets.dataloader_shuffle,
        num_workers=config.datasets.dataloader_num_workers,
        drop_last=True,
        pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_loader_denoising(config):
    dataset_train = Dataset_denoising_train(
        patch_size=config.datasets.train.image_size,
        sigma=config.sigma,
        input_channels=config.n_channels,
        H_path=config.datasets.train.H_path,
        L_path=config.datasets.train.L_path
    )
    dataset_val = Dataset_denoising_val(
        sigma=config.sigma,
        input_channels=config.n_channels,
        H_path=config.datasets.val.H_path,
        L_path=config.datasets.val.L_path
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.datasets.dataloader_batch_size,
        shuffle=config.datasets.dataloader_shuffle,
        num_workers=config.datasets.dataloader_num_workers,
        drop_last=True,
        pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_loader_deblur(config):
    dataset_train = Dataset_deblur_train(
        patch_size=config.datasets.train.image_size,
        input_channels=config.n_channels,
        H_path=config.datasets.train.H_path,
        L_path=config.datasets.train.L_path
    )
    dataset_val = Dataset_deblur_val(
        input_channels=config.n_channels,
        H_path=config.datasets.val.H_path,
        L_path=config.datasets.val.L_path
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.datasets.dataloader_batch_size,
        shuffle=config.datasets.dataloader_shuffle,
        num_workers=config.datasets.dataloader_num_workers,
        drop_last=True,
        pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val
