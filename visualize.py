import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import umap.umap_ as umap
from sklearn import manifold
from transformers import AutoTokenizer

from utils import get_verbalization_ids
from data_utils import PVPS

tokenizer = AutoTokenizer.from_pretrained(
    'roberta-large', cache_dir='pretrain/roberta-large', use_fast=False)


def analyze(res_list):
    loss_list = [res['eval_loss'] for res in res_list]
    acc_list = [res['scores']['acc'] for res in res_list]
    correct_logits, correct_ranks, mr, mrr = [], [], [], []
    mean_logits, std_logits = [], []
    for res in res_list:
        mr_tmp, mrr_tmp, logits_tmp, ranks_tmp = [], [], [], []
        for i in range(len(res['labels'])):
            label = res['labels'][i]
            sort_logit = sorted(res['full_logits'][i], reverse=True)
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for vid in verbalizer_ids[label]:
                logit = res['full_logits'][i][vid]
                rank = sort_logit.index(logit) + 1
                reciprocal = 1 / rank
                tmp1.append(rank)
                tmp2.append(reciprocal)
                tmp3.append(logit)
                tmp4.append(rank)
            mr_tmp.append(np.mean(tmp1))
            mrr_tmp.append(np.mean(tmp2))
            logits_tmp.append(np.mean(tmp3))
            ranks_tmp.append(np.mean(tmp4))
        correct_logits.append(logits_tmp)
        correct_ranks.append(ranks_tmp)
        mr.append(np.mean(mr_tmp))
        mrr.append(np.mean(mrr_tmp))
        mean_logits.append(np.mean(logits_tmp))
        std_logits.append(np.std(logits_tmp))
    return loss_list, acc_list, mr, mrr, mean_logits, std_logits


# What are high ranked words? How do they change?
# unused token reaches the 2 place
def get_top_words(res_list, verbalizers, k=10):
    top_words = []
    for res in res_list:
        curr_top_words = {label: {} for label in range(len(verbalizers))}
        curr_top_count = {label: [] for label in range(len(verbalizers))}
        for i in range(len(res['labels'])):
            label = res['labels'][i]
            # Only count first one
            indices = np.argsort(-res['full_logits'][i])[:k]
            words = tokenizer.convert_ids_to_tokens(indices)
            scores = res['full_logits'][i][indices]
            for word, score in zip(words, scores):
                if word not in curr_top_words[label]:
                    curr_top_words[label][word] = []
                curr_top_words[label][word].append(score)
        for label, word_dict in curr_top_words.items():
            for word, scores in word_dict.items():
                curr_top_count[label].append(
                    (word, len(scores), np.mean(scores)))
            curr_top_count[label] = sorted(
                curr_top_count[label], key=lambda p: (-p[1], -p[2]))[:k]
        top_words.append(curr_top_count)
    return top_words


# How do masked hidden states change?
def get_center(res_list, verbalizers):
    def calc_dist(hid1, hid2):
        return np.power(hid1 - hid2, 2).sum()

    centers, mean_dist, other_dist = [], [], []
    for res in res_list:
        curr_states = {label: [] for label in range(len(verbalizers))}
        for i in range(len(res['labels'])):
            label = res['labels'][i]
            hidden = res['masked_hidden_states'][i]
            curr_states[label].append(hidden)
        curr_center = {label: np.mean(hidden_list, axis=0)
                       for (label, hidden_list) in curr_states.items()}
        curr_dists, curr_other_dists = [], []
        for i in range(len(res['labels'])):
            label = res['labels'][i]
            hidden = res['masked_hidden_states'][i]
            curr_other_dists.append([])
            for iter_lab in range(len(verbalizers)):
                if iter_lab == label:
                    curr_dists.append(calc_dist(curr_center[label], hidden))
                else:
                    curr_other_dists[-1].append(
                        calc_dist(curr_center[iter_lab], hidden))
            curr_other_dists[-1] = np.mean(curr_other_dists[-1])
        centers.append(curr_center)
        mean_dist.append(np.mean(curr_dists))
        other_dist.append(np.mean(curr_other_dists))
    center_dist = []
    for i in range(len(centers) - 1):
        curr_dist = {}
        for label in range(len(verbalizers)):
            curr_dist[label] = calc_dist(
                centers[i + 1][label], centers[i][label])
        center_dist.append(curr_dist)
    return centers, mean_dist, other_dist, center_dist


def reduce_plot(res_list, fname, colors, reduce='tsne', n_dim=2):
    reducer_cls = {'tsne': manifold.TSNE, 'umap': umap.UMAP}
    nrows, ncols = 1, 4
    total = len(res_list)
    res_list = [res_list[0], res_list[total // 3],
                res_list[total // 3 * 2], res_list[-1]]
    fig = plt.figure(figsize=(ncols * 4, nrows * 4))
    gs = GridSpec(nrows, ncols, figure=fig)
    num_samples = len(res_list[0]['masked_hidden_states'])
    all_samples = np.concatenate([res['masked_hidden_states']
                                  for res in res_list], axis=0)
    reducer = reducer_cls[reduce](n_components=n_dim)
    embed = reducer.fit_transform(all_samples)
    for i in range(len(res_list)):
        labels = res_list[i]['labels']
        st, ed = i * num_samples, (i + 1) * num_samples
        if n_dim == 3:
            ax = fig.add_subplot(gs[i // ncols, i % ncols], projection='3d')
            ax.scatter(embed[st:ed, 0],
                       embed[st:ed, 1],
                       embed[st:ed, 2],
                       c=[colors[lab] for lab in labels])
        else:
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            ax.scatter(embed[st:ed, 0],
                       embed[st:ed, 1],
                       c=[colors[lab] for lab in labels])
    plt.savefig(fname)


def calc_dist(res_list):
    def calc_intra(hidden_):
        return (2 * len(hidden_) * np.sum(np.power(hidden_, 2)) - 2 * np.power(hidden_.sum(axis=0), 2).sum()) / (len(hidden_) ** 2)

    def calc_inter(hidden0_, hidden1_):
        n0, n1 = len(hidden0_), len(hidden1_)
        return (n1 * np.power(hidden0_, 2).sum() + n0 * np.power(hidden1_, 2).sum() - 2 * (np.sum(hidden0_, axis=0) * np.sum(hidden1_, axis=0)).sum()) / (n0 * n1)

    # total = len(res_list)
    # res_list = [res_list[0], res_list[total // 3],
    #             res_list[total // 3 * 2], res_list[-1]]
    # num_samples = len(res_list[0]['masked_hidden_states'])
    # all_samples = np.concatenate([res['masked_hidden_states']
    #                               for res in res_list], axis=0)
    # all_labels = np.concatenate([res['labels']
    #                             for res in res_list], axis=0)
    intra_list, inter_list = [], []
    for res in res_list:
        hidden = res['masked_hidden_states']
        labels = res['labels']
        hidden0, hidden1 = hidden[labels == 0], hidden[labels == 1]
        intra_list.append((calc_intra(hidden0), calc_intra(hidden1)))
        inter_list.append((calc_inter(hidden0, hidden1)))
    return intra_list, inter_list
    # MR
    # none, dev
    # [0.9161196877406708, 0.72374905100421, 0.6836791100209021, 0.5045967911736219, 0.3946192286793967, 0.3587972847457645, 0.3501557411236434, 0.34585884774364484, 0.3444828642831944, 0.3444616920343609, 0.34479935597429356, 0.3802596644703233, 0.6039614742048538, 0.6478383699252813, 0.6786934861584185, 0.6786934861584185]
    # none, test
    # [0.9227919069469269, 0.679849629778673, 0.4420182174570046, 0.4115889343670829, 0.403697712149834, 0.4011769244868452, 0.4011969765817195, 0.4016400965402332, 0.40217813839848987, 0.4025731523090457, 0.5065880336770153]

    # inner, dev
    # [0.9161196877406708, 0.7263132452465356, 0.6759360649318287, 0.4979943048266467, 0.3869553036641815, 0.3591734910706409, 0.35226267488097945, 0.3501359259088605, 0.3503131995350044, 0.35147671381616163, 0.35235660054067025, 0.37709984983207, 0.5997170676847495, 0.5875570146757654, 0.5791842050319007, 0.5791842050319007]
    # inner, test
    # [0.9227919069469269, 0.6815065588257899, 0.44005860053086215, 0.412474246895165, 0.4055451688555766, 0.40360208166132805, 0.40381080049843127, 0.4044275635704578, 0.40508328081282485, 0.40552716043328796, 0.3901748892305761, 0.4299200404122161]

    # CR
    # none, dev
    # [0.8648843992075033, 0.747585194275044, 0.6960699801484992, 0.6588624390770208, 0.63657800054905, 0.623374501676657, 0.6151051068444136, 0.6102884244264745, 0.6072472661450085, 0.6053776687468267, 0.604222760980095, 0.6032855262612054, 0.6026945452164825, 0.6023293009167406, 0.6023293009167406]
    # none, test
    # [0.9166192461758146, 0.7724505144384355, 0.6940862881098706, 0.636895213138533, 0.5544263851806871]

    # inner, dev
    # [0.8648843992075033, 0.741406341350502, 0.6078208251894295, 0.5611262783844687, 0.5635015650952776, 0.5894631660597703, 0.6115329655551883, 0.6242854943584383, 0.6317506811255532, 0.4904750723112895, 0.49866009012839185, 0.4623214830160259, 0.4539425887373832, 0.4539425887373832]
    # inner, test
    # [0.9166192461758146, 0.7615970884755277, 0.5900671600395057, 0.5127856034174797, 0.42717631938354983, 0.43386374239131603, 0.41496892118707907, 0.4064583815814711, 0.4064583815814711]


def dist_ratio_plot(res0, res1, fname):
    plt.figure(figsize=(6, 4))
    intra0, inter0 = calc_dist(res0)
    intra1, inter1 = calc_dist(res1)
    ratio0 = [np.mean(intra) / np.mean(inter)
              for (intra, inter) in zip(intra0, inter0)]
    ratio1 = [np.mean(intra) / np.mean(inter)
              for (intra, inter) in zip(intra1, inter1)]
    xs0 = np.linspace(0, 5 * len(intra0), num=len(intra0))
    xs1 = np.linspace(0, 5 * len(intra1), num=len(intra1))
    plt.plot(xs0, ratio0, 's-', color='r', label='Fixed')
    plt.plot(xs1, ratio1, 'o-', color='y', label='Differentiable')
    plt.xlabel('Training steps')
    plt.ylabel('Intra-class distance / inter-class distance')
    plt.legend(loc='best')
    plt.savefig(fname)


if __name__ == '__main__':
    task_name = 'cr'
    dev_res = pickle.load(
        open('visual/{}/{}-none-dev.eval'.format(task_name, task_name), 'rb'))
    inner_dev_res = pickle.load(
        open('visual/{}/{}-inner-dev.eval'.format(task_name, task_name), 'rb'))
    test_res = pickle.load(
        open('visual/{}/{}-none-test.eval'.format(task_name, task_name), 'rb'))
    inner_test_res = pickle.load(
        open('visual/{}/{}-inner-test.eval'.format(task_name, task_name), 'rb'))
    test_res = list(filter(None, test_res))
    inner_test_res = list(filter(None, inner_test_res))
    pvp = PVPS[task_name]
    verbalizers = list(pvp.VERBALIZER.values())
    verbalizer_ids = [[get_verbalization_ids(word, tokenizer, force_single_token=True) for word in words]
                      for words in verbalizers]

    print([a['scores']['acc'] for a in dev_res])
    print([a['scores']['acc'] for a in test_res])
    print('dev top words:', get_top_words(dev_res, verbalizers))
    print('test top words:', get_top_words(test_res, verbalizers))

    colors = ['r', 'deepskyblue', 'gold', 'g', 'black']
    reduce_plot(test_res, 'visual/{}/{}-none-test-tsne.pdf'.format(
        task_name, task_name), colors, 'tsne')
    reduce_plot(inner_test_res, 'visual/{}/{}-inner-test-tsne.pdf'.format(
        task_name, task_name), colors, 'tsne')
    dist_ratio_plot(dev_res, inner_dev_res,
                    'visual/{}/dist_ratio.pdf'.format(task_name))
