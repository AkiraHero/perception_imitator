import sys
sys.path.append('D:/1Pjlab/ModelSimulator/')
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
        sum_dict['have_fp'] = step_output['have_fp']
        sum_dict['all_fp'] = step_output['all_fp']
        sum_dict['hard_fp'] = step_output['hard_fp']


    else:
        sum_dict['have_fp'] = torch.cat([sum_dict['have_fp'], step_output['have_fp']], dim=0)
        sum_dict['all_fp'] = torch.cat([sum_dict['all_fp'], step_output['all_fp']], dim=0)
        sum_dict['hard_fp'] = torch.cat([sum_dict['hard_fp'], step_output['hard_fp']], dim=0)

def sum_output(gt_dict, step_output):
    if len(gt_dict.keys()) == 0:
        gt_dict.update(step_output)
    else:
        gt_dict['fp_prediction'] = torch.cat([gt_dict['fp_prediction'], step_output['fp_prediction']], dim=0)
        gt_dict['all_num_prediction'] = torch.cat([gt_dict['all_num_prediction'], step_output['all_num_prediction']], dim=0)
        gt_dict['hard_num_prediction'] = torch.cat([gt_dict['hard_num_prediction'], step_output['hard_num_prediction']], dim=0)


# todo: support 1.distributed testing 2.logger custimized for testing
if __name__ == '__main__':

    try:
        # # manage config
        # logging_logger = logging.getLogger()
        # logging_logger.setLevel(logging.NOTSET)

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
        ckpt = "D:/1Pjlab/ModelSimulator/output/fp_difficult/10.25.2(540)/540.pt"
        model.load_model_paras_from_file(ckpt)
        model.set_eval()
        model.set_device("cuda:0")
        output_data = {
            "predictions":[],
            "model_para_file": args.check_point_file
        }
        output_dir = "../output/fp_difficult"
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
                print("step:", step)

        
        # show fp prediction
        detected_truth = gt_summary['have_fp']
        detected_prediction = out_summary['fp_prediction']

        _, output_cls = detected_prediction.max(dim=1)
        # print(output_cls)

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        C = confusion_matrix(detected_truth.cpu(), output_cls.cpu(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=C)
        disp.plot()
        plt.title("confusion matrix: err on {}".format("fp_pred"))
        plt.savefig("confusion_mat-err-{}.png".format("fp_pred"))
        # plt.show()

        valid_inx = (gt_summary['have_fp'] == 1).nonzero()
        # show valid head
        # head_name = ['x', 'y', 'z', 'w', 'h', 'l', 'rot']
        # for inx, head in enumerate(head_name):
        #     target_ = gt_summary['box_diff_cls'][:, inx, :].long()
        #     _, cls_pred = out_summary['box_err_prediction'][head].max(dim=1)
        #     # screen the non-detected ones
        #     target_ = target_[valid_inx[:, 0], valid_inx[:, 1]]
        #     cls_pred = cls_pred[valid_inx[:, 0]]

        #     import matplotlib.pyplot as plt
        #     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        #     C = confusion_matrix(target_.cpu(), cls_pred.cpu(), normalize='true')
        #     disp = ConfusionMatrixDisplay(confusion_matrix=C)
        #     disp.plot()
        #     # plt.show()
        #     plt.title("confusion matrix: err on {}".format(head))
        #     plt.savefig("confusion_mat-err-{}.png".format(head))

        # show all fp number
        gt_all = gt_summary['all_fp'][valid_inx]
        print("all_fp_num:", gt_all.squeeze(1).cpu().numpy().tolist().count(0), gt_all.squeeze(1).cpu().numpy().tolist().count(1), gt_all.squeeze(1).cpu().numpy().tolist().count(2), gt_all.squeeze(1).cpu().numpy().tolist().count(3))
        _, pred_dt_all = out_summary['all_num_prediction'].max(dim=1)
        pred_dt_all = pred_dt_all[valid_inx]

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        C = confusion_matrix(gt_all.cpu(), pred_dt_all.cpu(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=C)
        disp.plot()
        plt.title("confusion matrix: err on {}".format("all_fp_number"))
        plt.savefig("confusion_mat-err-{}.png".format("all_fp_number"))
        # plt.show()

        # show  hard fp number
        gt_hard = gt_summary['hard_fp'][valid_inx]
        print("hard_fp_num:", gt_hard.squeeze(1).cpu().numpy().tolist().count(0), gt_hard.squeeze(1).cpu().numpy().tolist().count(1), gt_hard.squeeze(1).cpu().numpy().tolist().count(2), gt_hard.squeeze(1).cpu().numpy().tolist().count(3))
        _, pred_dt_hard = out_summary['hard_num_prediction'].max(dim=1)
        pred_dt_hard = pred_dt_hard[valid_inx]

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        C = confusion_matrix(gt_hard.cpu(), pred_dt_hard.cpu(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=C)
        disp.plot()
        plt.title("confusion matrix: err on {}".format("hard_fp_number"))
        plt.savefig("confusion_mat-err-{}.png".format("hard_fp_number"))
        plt.show()


        pass


    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
