from trainer.trainer_base import TrainerBase
from trainer.simple_trainer import SimpleTrainer
from trainer.vae_gan_trainer import VAEGANTrainer

class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(trainer_config):
        class_name = trainer_config['trainer_class']
        all_classes = TrainerBase.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(trainer_config)
        raise TypeError(f'no class named \'{class_name}\' found in trainer folder')