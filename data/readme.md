* gt_dt_matching_res.pkl包含了kitti原始gt数据、pvrcnn检测数据和配对情况，具体内容见亮哥给的邮件
* fp_bbox_data.pkl只存储了所有的fp检测框七个参数
* tpfp_explicit_data.pkl存储了所有pvrcnn检测结果中tp和fp的所在图片、该图片中的dtbox编号、11个参数(7个检测框、类别、两个点云反射数据和包围的点云个数)
* fp_bbox_data.pkl仅存储了FP的检测框的7个参数
* fp_difficult.pkl存储了每张图像中每个FP的难易度
* img_fp_difficult.pkl存储了每张图像是否拥有FP，拥有几个FP，FP中easy和hard的个数
* fp_cloud_point.pkl存储FP对应的点云数据及BBox7个参数