import os
import argparse
import yaml
import shutil

'''
treat the configuration as a tree
'''


class Configuration:
    def __init__(self):
        self.root_config = None
        self.expanded_config = None
        self.all_related_config_files = []

        # self.dataset_config = None
        # self.training_config = None
        # self.testing_config = None
        # self.logging_config = None
        # self.model_config = None
        pass

    def get_shell_args_train(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
        parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
        parser.add_argument('--epochs', type=int, default=None, required=False,
                            help='number of epochs to train for')
        parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
        parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
        parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
        parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
        parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
        parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
        parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                            help='set extra config keys if needed')
        parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
        parser.add_argument('--start_epoch', type=int, default=0, help='')
        parser.add_argument('--save_to_file', action='store_true', default=False, help='')
        args = parser.parse_args()
        return args

    def get_shell_args_test(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
        parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
        parser.add_argument('--epochs', type=int, default=None, required=False,
                            help='number of epochs to train for')
        parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
        parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
        parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
        parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
        parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
        parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
        parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                            help='set extra config keys if needed')
        parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
        parser.add_argument('--start_epoch', type=int, default=0, help='')
        parser.add_argument('--save_to_file', action='store_true', default=False, help='')
        args = parser.parse_args()
        return args

    @staticmethod
    def _load_yaml(file):
        with open(file, 'r') as f:
            return yaml.load(f)

    def load_config_file(self, config_file):
        self.root_config = Configuration._load_yaml(config_file)
        self.expanded_config = self.root_config.copy()
        self.all_related_config_files.append(config_file)
        self._expand_config(self.expanded_config)

    def _expand_config(self, config_dict):
        if not self._expand_cur_config(config_dict):
            if isinstance(config_dict, dict):
                for i in config_dict.keys():
                    sub_config = config_dict[i]
                    self._expand_config(sub_config)

    def _expand_cur_config(self, config_dict):
        if not isinstance(config_dict, dict):
            return False
        if 'config_file' in config_dict.keys() and isinstance(config_dict['config_file'], str):
            file_name = config_dict['config_file']
            expanded = Configuration._load_yaml(file_name)
            self.all_related_config_files.append(file_name)
            config_dict['config_file'] = {
                'file_name': file_name,
                'expanded': expanded
            }
            return True
        return False


    def pack_configurations(self, _path):
        # all config file should be located in utils/config?? no
        # pack config using expanded config
        pass


    def load_top_config_file(self, file):
        pass


    # only overwrite the first-found one on condition of equal keys
    def overwrite_config_by_shell_args(self, args):
        pass



