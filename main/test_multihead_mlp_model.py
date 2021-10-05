import logging
import pickle
import traceback
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
import torch
import os
import datetime


def sum_gt(sum_dict, step_output):
    if len(sum_dict.keys()) == 0:
        sum_dict['detected'] = step_output['detected']
        sum_dict['box_diff_cls'] = step_output['box_diff_cls']
        sum_dict['gt_box_cls'] = step_output['gt_box'][:, 7]
        sum_dict['dt_box_cls'] = step_output['dt_box'][:, 7]

    else:
        sum_dict['detected'] = torch.cat([sum_dict['detected'], step_output['detected']], dim=0)
        sum_dict['box_diff_cls'] = torch.cat([sum_dict['box_diff_cls'], step_output['box_diff_cls']], dim=0)
        sum_dict['gt_box_cls'] = torch.cat([sum_dict['gt_box_cls'], step_output['gt_box'][:, 7]], dim=0)
        sum_dict['dt_box_cls'] = torch.cat([sum_dict['dt_box_cls'], step_output['dt_box'][:, 7]], dim=0)


def sum_output(gt_dict, step_output):
    if len(gt_dict.keys()) == 0:
        gt_dict.update(step_output)
    else:
        gt_dict['fn_prediction'] = torch.cat([gt_dict['fn_prediction'], step_output['fn_prediction']], dim=0)
        gt_dict['cls_prediction'] = torch.cat([gt_dict['cls_prediction'], step_output['cls_prediction']], dim=0)
        for name in gt_dict['box_err_prediction'].keys():
            gt_dict['box_err_prediction'][name] = \
                torch.cat([gt_dict['box_err_prediction'][name], step_output['box_err_prediction'][name]], dim=0)

# todo: support 1.distributed testing 2.logger custimized for testing
if __name__ == '__main__':

    try:
        # manage config
        logging_logger = logging.getLogger()
        logging_logger.setLevel(logging.NOTSET)

        config = Configuration()
        args = config.get_shell_args_train()

        # default test settings
        args.for_train = False
        args.shuffle = False

        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
        data_loader = dataset.get_data_loader()

        model = ModelFactory.get_model(config.model_config)
        ckpt = "/home/akira/Project/ModelSimulator/output/2021-10-05-18-55-03-only_box/model_paras_log/model_ckpt-epoth950-step10-2021-10-05-19-02-37.pt"
        model.load_model_paras_from_file(ckpt)
        model.set_eval()
        model.set_device("cuda:0")
        output_data = {
            "predictions":[],
            "model_para_file": args.check_point_file
        }
        output_dir = "../output"
        date_time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        output_file = os.path.join(output_dir, "test_log-" + date_time_str + '.pkl')
        out_summary = {}
        gt_summary = {}
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                dataset.load_data2gpu(data)
                out = model(data)
                sum_output(out_summary, out)
                sum_gt(gt_summary, data)
                pass

        # show fp prediction
        detected_truth = gt_summary['detected']
        detected_prediction = out_summary['fn_prediction']
        output_cls = torch.zeros(detected_prediction.shape, device=detected_prediction.device)
        detected_inx = (detected_prediction > 0.5).nonzero()
        output_cls[detected_inx] = 1
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        C = confusion_matrix(detected_truth.cpu(), output_cls.cpu(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=C)
        disp.plot()
        # plt.show()
        plt.title("confusion matrix: err on {}".format("fn_pred"))
        plt.savefig("confusion_mat-err-{}.png".format("fn_pred"))

        valid_inx = (gt_summary['detected'] == 1).nonzero()
        # show valid head
        head_name = ['x', 'y', 'z', 'w', 'h', 'l', 'rot']
        for inx, head in enumerate(head_name):
            target_ = gt_summary['box_diff_cls'][:, inx, :].long()
            _, cls_pred = out_summary['box_err_prediction'][head].max(dim=1)
            # screen the non-detected ones
            target_ = target_[valid_inx[:, 0], valid_inx[:, 1]]
            cls_pred = cls_pred[valid_inx[:, 0]]

            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            C = confusion_matrix(target_.cpu(), cls_pred.cpu(), normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=C)
            disp.plot()
            # plt.show()
            plt.title("confusion matrix: err on {}".format(head))
            plt.savefig("confusion_mat-err-{}.png".format(head))

        # show class distribution
        gt_class = gt_summary['gt_box_cls'][valid_inx[:, 0]]
        target_dt_class = gt_summary['dt_box_cls'][valid_inx[:, 0]]
        _, pred_dt_class = out_summary['cls_prediction'].max(dim=1)
        # label 0 denote invalid in "target_dt_class", which means non-detected
        # so 0 <= target_dt_class <=3,  1 <= gt_class <= 3, 0 <= pred_dt_class <= 2
        pred_dt_class = pred_dt_class[valid_inx[:, 0]] + 1

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        C = confusion_matrix(gt_class.cpu(), target_dt_class.cpu(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=C)
        disp.plot()
        # plt.show()
        plt.title("confusion matrix: err on {}".format("dt class"))
        plt.savefig("confusion_mat-err-{}.png".format("dt_class"))

        C = confusion_matrix(target_dt_class.cpu(), pred_dt_class.cpu(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=C)
        disp.plot()
        # plt.show()
        plt.title("confusion matrix: err on {}".format("pred_dt class"))
        plt.savefig("confusion_mat-err-{}.png".format("pred_dt_class"))


        pass


    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)