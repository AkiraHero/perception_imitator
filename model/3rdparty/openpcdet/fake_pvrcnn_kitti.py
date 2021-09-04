# add pvrcnn pretrained model results file and its gt matching file
import os
import pickle
from model.model_base import ModelBase
from multiprocessing import Process

class FakePVRCNNOnKitti(ModelBase):
    def __init__(self, config):
        super(FakePVRCNNOnKitti, self).__init__()
        self.kitti_results_folder = config['kitti_results_folder']
        self.num_workers = config['num_workers']

    def _get_result(self, frm_id, res_dict):
        results_file = os.path.join(self.kitti_results_folder, f'{frm_id}.pkl')
        res = None
        with open(results_file, 'r') as f:
            res = pickle.load(f)
            # todo: is it safe to write dict in multiprocessing?
            res_dict[frm_id] = res

    def _get_results(self, frm_id_unit, res_dict):
        for i in frm_id_unit:
            self._get_result(i, res_dict)

    def _split_ids(self, ids):
        work_size = len(ids)
        if work_size < self.num_workers:
            self.num_workers = work_size
        work_unit_size = work_size // self.num_workers
        work_unit_list = []
        idx = 0
        for i in range(self.num_workers):
            work_unit = ids[idx:idx + work_unit_size]
            work_unit_list.append(work_unit)
            idx = idx + work_unit_size
        return work_unit_list

    def _get_batch_result(self, data_batch):
        result_dict = {}
        ids = data_batch['frame_id']
        id_units = self._split_ids(ids)
        proc_list = []
        for unit in id_units:
            p = Process(self._get_results, unit, result_dict)
            proc_list.append(p)
            p.start()
        for p in proc_list:
            p.join()
        return result_dict

    def forward(self, data_dict):
        return self._get_batch_result(data_dict)