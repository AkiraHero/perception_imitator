from unittest.test_trivial_purebox.boxonly_dataset import *
from analysis.model_error_analysis.get_detection3d_error_stats import plot_error_statistics


dataset = SimpleDataset(load_all=True, batch_size=100000)
data_loader = dataset.get_data_loader()
for step, data in enumerate(data_loader):
    f = plot_error_statistics(data['gt_box'], data['dt_box'])
    f.savefig("kitti_instance_error_pvrcnn.png")
    break
pass