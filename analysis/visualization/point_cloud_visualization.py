import ctypes
import os
from ctypes import cdll
from functools import partial
from scipy.spatial.transform import Rotation
import numpy as np

class PointCloudVisualization:
    def __init__(self):
        ll = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libvisualization_tools.so"))
        ll.get_pcl_vis_instance.restype = ctypes.c_void_p
        self._c_pcl_visualizer = ctypes.c_void_p(ll.get_pcl_vis_instance())
        self._load_kitti_bin_func = partial(ll.load_kitti_bin, self._c_pcl_visualizer)
        self._add_point_cloud_func = partial(ll.add_point_cloud, self._c_pcl_visualizer)
        self._show_func = partial(ll.show, self._c_pcl_visualizer)
        self._new_p_func = ll.new_pointcloud_xyz_ptr
        self._new_p_func.restype = ctypes.c_void_p
        self._close_func = partial(ll.stop_show, self._c_pcl_visualizer)
        self._add_box_3d_func = partial(ll.add_bounding_box_3d, self._c_pcl_visualizer)
        self._clear_all_fuc = partial(ll.clear_all, self._c_pcl_visualizer)

    def load_kitti_bin(self, file_name):
        empty_point_cloud_ptr = ctypes.c_void_p(p._new_p_func())
        s = ctypes.c_char_p(file_name.encode())
        self._load_kitti_bin_func(s, empty_point_cloud_ptr)
        return empty_point_cloud_ptr

    def add_point_cloud(self, c_cloud_obj, str_id="cloud"):
        self._add_point_cloud_func(c_cloud_obj, ctypes.c_char_p(str_id.encode()))

    def show(self):
        self._show_func()

    def close(self):
        self.close_func()

    def add_bounding_box_3d(self, box_list, color_list, str_id_list):
        if not (isinstance(box_list, list)
                and isinstance(color_list, list) and isinstance(str_id_list, list)):
            raise TypeError
        if len(box_list) != len(color_list) \
                or len(box_list) != len(str_id_list):
            raise TypeError
        for box, color, id_ in zip(box_list, color_list, str_id_list):
            # todo: add colors
            rot = box[6]
            loc_c = (ctypes.c_float * 3)(*box[:3])
            dim_c = (ctypes.c_float * 3)(*box[3:6])
            r = Rotation.from_euler('zyx', [0, rot, 0], degrees=False)
            q = r.as_quat()
            quaternion_c = (ctypes.c_float * 4)(*q)
            self._add_box_3d_func(loc_c, dim_c, quaternion_c, ctypes.c_char_p(id_.encode()))

    def clear_all(self):
        pass

    # def load_kitti_label(self, label_file):
    #     with open(label_file) as f:
    #         lines = f.readlines(label_file)
    #         for line in lines:
    #             item_dict = {
    #
    #             }
    #             segments = line.split(" ")
    #             if segments[0] == 'Car':
    #                 pass
    #             else:
    #                 continue


if __name__ == '__main__':
    p = PointCloudVisualization()
    cloud_ptr = p.load_kitti_bin("/home/akira/000004.bin")
    p.add_point_cloud(cloud_ptr)
    tst_box_3d = [[10, 15, -5, 6, 8, 10, 0.6]]
    colors = [[255, 0, 0]]
    names = ["hh1"]
    p.add_bounding_box_3d(tst_box_3d, colors, names)
    tst_box_3d = [[20, -15, -5, 6, 8, 10, 0.6]]
    names = ["hh2"]

    p.add_bounding_box_3d(tst_box_3d, colors, names)
    p.show()
