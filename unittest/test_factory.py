from factory.dataset_factory import DatasetFactory


data_config = {}
data_config['dataset_class'] = 'MinistDataset'

loader = DatasetFactory.get_data_loader(data_config)
pass