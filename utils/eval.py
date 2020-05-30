import os
import pickle
import numpy as np
#import matplotlib.pyplot as plt


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
    apcer = fp / len(ind_n) * 100
    bpcer = fn / len(ind_p) * 100
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
    ind_n = (results[:, 1] == 0)
    ind_p = (results[:, 1] == 1)
    far = (results[ind_n, 0] == 1).sum()
    frr = (results[ind_p, 0] == 0).sum()
    far = far / len(ind_n) * 100
    frr = frr / len(ind_p) * 100
    hter = (far + frr) / 2
    if is_print:
        print('***************************************')
        print(' FAE     FRR     HTRE')
        print('{:.4f}   {:.4f}   {:.4f}'.format(far, frr, hter))
        print('***************************************')
    return 100 - hter


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
    if type not in ['acc', 'acer', 'hter']:
        raise NotImplementedError
    else:
        eval_tool = eval_tools[type]
    results = np.array(results)

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


#def Vis_Results(results, result_dir):
#    living = []
#    spoofing = []
#    for res in results:
#        if int(res[1]) == 1:
#            living.append(res[2])
#        else:
#            spoofing.append(res[2])
#    minb = min(living+spoofing)
#    maxb = max(living+spoofing)
#    bins = np.linspace(minb, maxb, 80)
#    plt.figure(1)
#    plt.title(' ')
#    plt.hist(living, bins, facecolor='g', edgecolor="black", alpha=0.5, label='living')
#    plt.hist(spoofing, bins, facecolor='r', edgecolor="black", alpha=0.5, label='spoofing')
#    plt.legend()
#    plt.xlabel('Threshold')
#    plt.ylabel('Number Sample')
#    plt.savefig(os.path.join(result_dir, 'VR4@1_test.png'))
#    # plt.show()

