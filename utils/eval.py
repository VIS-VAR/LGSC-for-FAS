import os
import pickle
import numpy as np
from .util import draw_roc
from .statistic import get_EER_states, get_HTER_at_thr
from sklearn.metrics import roc_auc_score


def eval_acer(results, is_print=False):
    """
    :param results: np.array shape of (N, 2) [pred, label]
    :param is_print: print eval score
    :return: score
    """
    ind_n = (results[:, 1] == 0)
    ind_p = (results[:, 1] == 1)
    fp = (results[ind_n, 0] == 1).sum()
    fn = (results[ind_p, 0] == 0).sum()
    apcer = fp / ind_n.sum() * 100
    bpcer = fn / ind_p.sum() * 100
    acer = (apcer + bpcer) / 2
    if is_print:
        print('***************************************')
        print('APCER    BPCER     ACER')
        print('{:.4f}   {:.4f}   {:.4f}'.format(apcer, bpcer, acer))
        print('***************************************')
    return 100 - acer


def eval_hter(results, is_print=False):
    """
    :param results: np.array shape of (N, 2) [pred, label]
    :param is_print: print eval score
    :return: score
    """
    prob_list = results[:, 0]
    label_list = results[:, 1]
    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    auc_score = roc_auc_score(label_list, prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    if is_print:
        print('***************************************')
        print('EER        HTER      AUC        Thr')
        print('{:.4f}   {:.4f}   {:.4f}    {:.4f}'.format(
            cur_EER_valid * 100, cur_HTER_valid * 100, auc_score * 100, threshold))
        print('***************************************')
    return (1 - cur_HTER_valid) * 100


def eval_acc(results, is_print=False):
    """
    :param results: np.array shape of (N, 2) [pred, label]
    :param is_print: print eval score
    :return: score
    """
    acc = (results[:, 0] == results[:, 1]).sum() / results.shape[0] * 100
    if is_print:
        print('*****************')
        print('ACC   Pos')
        print('{:.2f}  {}'.format(acc, int(results[:, 0].sum())))
        print('*****************')
    return acc


def eval_metric(results, thr='auto', type='acc', res_dir=None):
    """
    :param results: np.array shape of (N, 2) [pred, label]
    :param type: acc acer  or hter
    :param res_dir: save eval results
    :return: best score
    """
    eval_tools = dict(
        acc=eval_acc,
        acer=eval_acer,
        hter=eval_hter)
    results = np.array(results)
    if type not in ['acc', 'acer', 'hter']:
        raise NotImplementedError
    elif type == 'hter':
        eval_score = eval_hter(results, is_print=True)
        return eval_score
    else:
        eval_tool = eval_tools[type]

    if isinstance(thr, float):
        results[:, 0] = (results[:, 0] > thr).astype(np.float)
        results = results.astype(np.int)
        return eval_tool(results, is_print=True)

    min_score = results[:, 0].min()
    max_score = results[:, 0].max()
    s_step = (max_score - min_score) / 1000
    scores = []
    thrs = []
    for i in range(1000):
        thre = min_score + i * s_step
        thrs.append(thre)
        result = results.copy()
        result[:, 0] = (results[:, 0] > thre).astype(np.float)
        result = result.astype(np.int)
        score = eval_tool(result, is_print=False)
        scores.append(score)
    max_ind = np.argmax(np.array(scores))
    if thr == 'mid':
        sinds = np.argsort(results[:, 0])
        best_thr = results[sinds[int(results.shape[0]/2)-1], 0]
    else:
        best_thr = thrs[max_ind]
    print('Best Threshold: {:.4f}'.format(best_thr))
    save_results = np.zeros((results.shape[0], 3))
    save_results[:, 2] = results[:, 0]
    results[:, 0] = (results[:, 0] > best_thr).astype(np.float)
    save_results[:, :2] = results[:, :2]
    eval_score = eval_tool(results, is_print=True)
    if res_dir is not None:
        res_dir = os.path.join(res_dir, 'res_{}.pkl'.format(int(eval_score * 10)))
        with open(res_dir, 'wb') as file:
            pickle.dump(save_results, file)
    return eval_score



