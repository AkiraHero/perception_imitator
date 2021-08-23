from utils.config.Configuration import Configuration
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from trainer.TrainerFactory import TrainerFactory


if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    config.load_config_file()
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    model = ModelFactory.get_model(config.model_config)
    dataset = DatasetFactory.get_data_loader(config.dataset_config)
    trainer = TrainerFactory.get_trainer(config.training_config)

    trainer.set(model=model, dataset=dataset)
    trainer.run()
