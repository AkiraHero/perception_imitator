from trainer.trainer_base import TrainerBase


class SimpleTrainer(TrainerBase):
    def __init__(self):
        super(SimpleTrainer, self).__init__()

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")

