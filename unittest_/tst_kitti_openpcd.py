from utils.config.Configuration import Configuration
from factory.dataset_factory import DatasetFactory




if __name__ == '__main__':
    config = Configuration()
    args = config.get_shell_args_train()
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    pass