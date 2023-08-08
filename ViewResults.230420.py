from pathlib import Path
import pandas as pd
import glob
import torch
import re
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
import umap
from torch.utils.tensorboard import SummaryWriter
import plotly.express as px

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', './results/rinna-japanese-gpt2-medium', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', './results/bert-base-japanese-whole-word-masking', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', './results/mbart-large-cc25', 100
    elif mode == 't5-base-encoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web', 100
    elif mode == 't5-base-decoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', './results/rinna-japanese-roberta-base', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', './results/nlp-waseda-roberta-base-japanese', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', './results/nlp-waseda-roberta-large-japanese', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', './results/rinna-japanese-gpt-1b', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', './results/xlm-roberta-large', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', './results/xlm-roberta-base', 100

def gather_details_by_categories(data_type):
    output_file = f'total_list.{data_type}.230528.0.csv'

    rets = []
    num_seed = 1
    num_fold = 5
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    list_metrics = ['f1', 'precision', 'recall']
    METHOD_LIST ={'rand': ['.230413', '.rand'],
                'eos': ['.230413', '.eos'],
                'cls': ['.230301', '.cls'],
                'lconcat': ['.230227', '.concat'],
                'gconcat': ['.230418', '.gconcat'],
                'conj': ['.230507', '.conj'],
                'wonone_conj': ['.230430', '.wonone.conj'],
                'wonone_cls': ['.230502', '.wonone.cls'],
                'conj_init': ['.230311', '.conj.init'],
                'conj_twostage': ['.230518', '.twostage.conj'],
                'conjcls_twostage': ['.230518', '.twostage.conjcls']}

    # METHOD_LIST ={'conj': ['.230223', '.conj'], 'wonone_conj': ['.230430', '.wonone.conj']}

    # MODE_LIST = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    MODE_LIST = ['xlm-roberta-large']

    sentence_pairs = {}
    golds = {}
    preds = {}
    info = {}
    methods = []
    for method, v in METHOD_LIST.items():
        methods.append(method)

        TAG = v[0]
        METHOD = v[1]

        for mode in MODE_LIST:
            model_name, OUTPUT_PATH, _ = get_properties(mode)
            OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
            for iseed in range(num_seed):
                for ifold in range(num_fold):
                    files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                    for items in torch.load(files[0])[f'{data_type}_results']:
                        key = f'{items["id"]}-{items["id_s1"]}-{items["id_s2"]}'
                        if key not in sentence_pairs:
                            sentence_pairs[key] = [items['s1'], items['s2']]
                        else:
                            pass

                        if key not in golds:
                            golds[key] = items['reason']
                        else:
                            pass

                        if key not in preds:
                            preds[key] = [items['predicted_label']]
                        else:
                            preds[key].append(items['predicted_label'])
                        
                        if key not in info:
                            info[key] = [f'{ifold}-{iseed}']
                        else:
                            if f'{ifold}-{iseed}' not in set(info[key]):
                                info[key].append(f'{ifold}-{iseed}')

        for k in preds.keys():
            if len(preds[k]) != len(methods):
                preds[k].append('')


    data = [[key, '|'.join(info[key]), sentence_pairs[key][0], sentence_pairs[key][1], golds[key]] + preds[key] for key in sentence_pairs.keys()]
    df = pd.DataFrame(data=data, columns=['id', 'fold-seed', 's1', 's2', 'gold']+methods, index=None)
    df.to_csv(output_file, encoding='utf-8')

    return rets

def gather_details_by_categories_avg_by_seed(data_type):
    output_file = f'total_list.{data_type}.avgbyseed.230628.0.csv'

    rets = []
    num_seed = 3
    num_fold = 5
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接', 'その他根拠', '談話関係なし'])}
    list_metrics = ['f1', 'precision', 'recall']
    METHOD_LIST ={'rand': ['.230413', '.rand'],
                'eos': ['.230413', '.eos'],
                'cls': ['.230301', '.cls'],
                'lconcat': ['.230227', '.concat'],
                'gconcat': ['.230418', '.gconcat'],
                'conj': ['.230507', '.conj'],
                'wonone_conj': ['.230430', '.wonone.conj'],
                'wonone_cls': ['.230502', '.wonone.cls']}

    # METHOD_LIST ={'conj': ['.230223', '.conj'], 'wonone_conj': ['.230430', '.wonone.conj']}

    # MODE_LIST = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    MODE_LIST = ['xlm-roberta-large']

    sentence_pairs = {}
    golds = {}
    preds = {}
    info = {}
    methods = []
    for method, v in METHOD_LIST.items():
        print(method)
        methods.append(method)

        TAG = v[0]
        METHOD = v[1]

        for mode in MODE_LIST:
            logits = {}
            model_name, OUTPUT_PATH, _ = get_properties(mode)
            OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
            for ifold in range(num_fold):
                for iseed in range(num_seed):
                    files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                    file = torch.load(files[0])
                    for items in file[f'{data_type}_results']:
                        key = f'{items["id"]}-{items["id_s1"]}-{items["id_s2"]}'
                        if key not in sentence_pairs:
                            sentence_pairs[key] = [items['s1'], items['s2']]
                        else:
                            pass

                        if key not in golds:
                            golds[key] = items['reason']
                        else:
                            pass

                        # if f'{key}-{iseed}' not in preds:
                        #     preds[f'{key}-{iseed}'] = [items['predicted_label']]
                        # else:
                        #     preds[f'{key}-{iseed}'].append(items['predicted_label'])

                        if key not in logits:
                            logits[key] = [torch.softmax(torch.as_tensor(items['logits']), dim=0).tolist()]
                        else:
                            logits[key].append(torch.softmax(torch.as_tensor(items['logits']), dim=0).tolist())

                        if key not in info:
                            info[key] = [f'{ifold}']
                        else:
                            if f'{ifold}' not in set(info[key]):
                                info[key].append(f'{ifold}')

            label_indexer = file['label_indexer']
            rev_label_indexer = {v: k for k, v in label_indexer.items()}
            for k in logits.keys():
                if k not in preds:
                    preds[k] = [rev_label_indexer[np.argmax(np.mean(logits[k], axis=0))]]
                else:
                    preds[k].append(rev_label_indexer[np.argmax(np.mean(logits[k], axis=0))])

        for k in preds.keys():
            if len(preds[k]) != len(methods):
                preds[k].append('')


    data = [[key, '|'.join(info[key]), sentence_pairs[key][0], sentence_pairs[key][1], golds[key]] + preds[key] for key in sentence_pairs.keys()]
    df = pd.DataFrame(data=data, columns=['id', 'fold', 's1', 's2', 'gold']+methods, index=None)
    df.to_csv(output_file, encoding='utf-8')

    return rets

def gather_embeddings(data_type='test'):
    # output_file = f'total_list.{data_type}.avgbyseed.230628.0.csv'

    rets = []
    num_seed = 1
    num_fold = 1
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接', 'その他根拠', '談話関係なし'])}
    # list_metrics = ['f1', 'precision', 'recall']
    METHOD_LIST ={'rand': ['.230413', '.rand'],
                'eos': ['.230413', '.eos'],
                'cls': ['.230301', '.cls'],
                'lconcat': ['.230227', '.concat'],
                'gconcat': ['.230418', '.gconcat'],
                'conj': ['.230507', '.conj']}
                # 'conj_init': ['.230311', '.conj.init']}
                # 'wonone_conj': ['.230430', '.wonone.conj'],
                # 'wonone_cls': ['.230502', '.wonone.cls']}
    MODE_LIST = ['xlm-roberta-large']

    methods = []
    for method, v in METHOD_LIST.items():
        print(method)
        methods.append(method)

        TAG = v[0]
        METHOD = v[1]

        for mode in MODE_LIST:
            sentence_pairs = {}
            golds = {}
            preds = {}
            info = {}
            last_hidden_states = {}
            model_name, OUTPUT_PATH, _ = get_properties(mode)
            OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
            for ifold in range(num_fold):
                for iseed in range(num_seed):
                    files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                    file = torch.load(files[0])
                    for items in file[f'{data_type}_results']:
                        key = f'{items["id"]}-{items["id_s1"]}-{items["id_s2"]}'

                        if key not in sentence_pairs:
                            sentence_pairs[key] = [items['s1'], items['s2']]
                        else:
                            pass

                        if key not in golds:
                            golds[key] = items['reason']
                        else:
                            pass

                        if key not in last_hidden_states:
                            if method == 'cls':
                                last_hidden_states[key] = [torch.as_tensor(items['last_hidden_state'][0]).tolist()]
                            else:
                                last_hidden_states[key] = [torch.as_tensor(items['last_hidden_state']).tolist()]
                        else:
                            if method == 'cls':
                                last_hidden_states[key].append(torch.as_tensor(items['last_hidden_state'][0]).tolist())
                            else:
                                last_hidden_states[key].append(torch.as_tensor(items['last_hidden_state']).tolist())

                        if key not in info:
                            info[key] = [f'{ifold}']
                        else:
                            if f'{ifold}' not in set(info[key]):
                                info[key].append(f'{ifold}')

            label_indexer = file['label_indexer']
            rev_label_indexer = {v: k for k, v in label_indexer.items()}
            for k in last_hidden_states.keys():
                if k not in preds:
                    preds[k] = [np.mean(last_hidden_states[k], axis=0).tolist()]
                else:
                    preds[k].append(np.mean(last_hidden_states[k], axis=0).tolist())


            dict_label = {'原因・理由': '原因・理由', '目的': '目的', '条件': '条件', '根拠': '根拠', '対比': '対比', '逆接': '逆接・譲歩', 'その他根拠': 'その他根拠', '談話関係なし': '談話関係なし'}
            dict_label_en = {'原因・理由': 'CAUSE/REASON', '目的': 'PURPOSE', '条件': 'CONDITION', '根拠': 'JUSTIFICATION', '対比': 'CONTRAST', '逆接・譲歩': 'CONCESSION', 'その他根拠': 'MISC', '談話関係なし': 'NONE'}
            embeddings = [v[0] for v in preds.values()]
            labels = [dict_label_en[dict_label[golds[k]]] for k in preds.keys()]

            embeddings = [e for e, l in zip(embeddings, labels) if l != 'NONE']
            labels = [l for l in labels if l != 'NONE']

            mode_decomposition = 'pca'
            if mode_decomposition == 'pca':
                pca = PCA(n_components=2, random_state=0)
                pca.fit(embeddings)
                reduced_embedding = pca.transform(embeddings) # {k: pca.transform(embeddings[k]).compute() for k in labels}
            elif mode_decomposition == 'tsne':
                tsne = TSNE(n_components=2, random_state=0)
                reduced_embedding = tsne.fit_transform(np.array(embeddings))
            elif mode_decomposition == 'umap':
                mapper = umap.UMAP(random_state=0)
                reduced_embedding = mapper.fit_transform(np.array(embeddings))

            pca_X = reduced_embedding.tolist()
            pca_Y = labels
            data = [x + [y] for x, y in zip(pca_X, pca_Y)]
            if mode_decomposition == 'pca':
                col1, col2 = 'PC1', 'PC2'
            elif mode_decomposition == 'tsne':
                col1, col2 = 'TSNE1', 'TSNE2'
            elif mode_decomposition == 'umap':
                col1, col2 = 'UMAP1', 'UMAP2'

            hue_order = ['CAUSE/REASON', 'CONCESSION', 'CONDITION', 'PURPOSE', 'JUSTIFICATION', 'CONTRAST']
            data = pd.DataFrame(data=data, columns=[col1, col2, 'category'])
            plt.clf()
            plt.figure()
            g = sns.scatterplot(data=data, x=col1, y=col2, hue='category', hue_order=hue_order, palette='Set1')
            plt.legend(fontsize='xx-small', title='Category', title_fontsize='xx-small')
            plt.savefig(f'./test.scatterplot.{model_name.replace("/", ".")}.230806.{mode_decomposition}{METHOD}{TAG}.png')
            # fig = px.scatter(data, x=col1, y=col2, color='category')
            # fig.write_image(f'./scatterplot.{model_name.replace("/", ".")}.230806.{METHOD}{TAG}.{mode_decomposition}.png')

def gather_embeddings_by_cases(data_type='test'):
    # output_file = f'total_list.{data_type}.avgbyseed.230628.0.csv'

    rets = []
    num_seed = 1
    num_fold = 1
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接', 'その他根拠', '談話関係なし'])}
    # list_metrics = ['f1', 'precision', 'recall']
    METHOD_LIST ={'rand': ['.230413', '.rand'],
                'eos': ['.230413', '.eos'],
                'cls': ['.230301', '.cls'],
                'lconcat': ['.230227', '.concat'],
                'gconcat': ['.230418', '.gconcat'],
                'conj': ['.230507', '.conj']}
                # 'conj_init': ['.230311', '.conj.init']}
                # 'wonone_conj': ['.230430', '.wonone.conj'],
                # 'wonone_cls': ['.230502', '.wonone.cls']}
    MODE_LIST = ['xlm-roberta-large']

    with Path('./data/datasets.230201/disc_kwdlc/implicitlist.txt').open('r') as f:
        list_implicit_instances = f.readlines()
        list_implicit_instances = set([item.strip() for item in list_implicit_instances])

    with Path('./data/datasets.230201/disc_kwdlc/backwardlist.txt').open('r') as f:
        list_backward_instances = f.readlines()
        list_backward_instances = set([item.strip() for item in list_backward_instances])

    methods = []
    for method, v in METHOD_LIST.items():
        print(method)
        methods.append(method)

        TAG = v[0]
        METHOD = v[1]

        for mode in MODE_LIST:
            sentence_pairs = {}
            golds = {}
            preds = {}
            info = {}
            last_hidden_states = {}
            model_name, OUTPUT_PATH, _ = get_properties(mode)
            OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
            for ifold in range(num_fold):
                for iseed in range(num_seed):
                    files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                    file = torch.load(files[0])
                    for items in file[f'{data_type}_results']:
                        key = f'{items["id"]}-{items["id_s1"]}-{items["id_s2"]}'

                        if key not in sentence_pairs:
                            sentence_pairs[key] = [items['s1'], items['s2']]
                        else:
                            pass

                        if key not in golds:
                            golds[key] = items['reason']
                        else:
                            pass

                        if key not in last_hidden_states:
                            if method == 'cls':
                                last_hidden_states[key] = [torch.as_tensor(items['last_hidden_state'][0]).tolist()]
                            else:
                                last_hidden_states[key] = [torch.as_tensor(items['last_hidden_state']).tolist()]
                        else:
                            if method == 'cls':
                                last_hidden_states[key].append(torch.as_tensor(items['last_hidden_state'][0]).tolist())
                            else:
                                last_hidden_states[key].append(torch.as_tensor(items['last_hidden_state']).tolist())

                        if key not in info:
                            info[key] = [f'{ifold}']
                        else:
                            if f'{ifold}' not in set(info[key]):
                                info[key].append(f'{ifold}')

            label_indexer = file['label_indexer']
            rev_label_indexer = {v: k for k, v in label_indexer.items()}
            for k in last_hidden_states.keys():
                if k not in preds:
                    preds[k] = [np.mean(last_hidden_states[k], axis=0).tolist()]
                else:
                    preds[k].append(np.mean(last_hidden_states[k], axis=0).tolist())


            dict_label = {'原因・理由': '原因・理由', '目的': '目的', '条件': '条件', '根拠': '根拠', '対比': '対比', '逆接': '逆接・譲歩', 'その他根拠': 'その他根拠', '談話関係なし': '談話関係なし'}
            dict_label_en = {'原因・理由': 'CAUSE/REASON', '目的': 'PURPOSE', '条件': 'CONDITION', '根拠': 'JUSTIFICATION', '対比': 'CONTRAST', '逆接・譲歩': 'CONCESSION', 'その他根拠': 'MISC', '談話関係なし': 'NONE'}
            # if key not in list_implicit_instances: # implicit
            # if key in list_implicit_instances: # explicit
            # if key not in list_backward_instances: # backward
            # if key in list_backward_instances: # forward
            #     continue

            embeddings = [v[0] for v in preds.values()]
            # labels = [dict_label_en[dict_label[golds[k]]] for k in preds.keys()]
            labels = ['Implicit - ' + dict_label_en[dict_label[golds[k]]] if k in list_implicit_instances else 'Explicit - ' + dict_label_en[dict_label[golds[k]]] for k in preds.keys()]
            labels = [l if 'CONCESSION' in l else l.replace('Implicit - ', '').replace('Explicit - ', '') for l in labels]
            eximplicits = ['Implicit' if k in list_implicit_instances else 'Explicit' for k in preds.keys()]
            sentences = [sentence_pairs[k] for k in preds.keys()]

            mode_decomposition = 'pca'
            if mode_decomposition == 'pca':
                pca = PCA(n_components=2)
                pca.fit(embeddings)
                reduced_embedding = pca.transform(embeddings) # {k: pca.transform(embeddings[k]).compute() for k in labels}
            elif mode_decomposition == 'tsne':
                tsne = TSNE(n_components=2, random_state=0)
                reduced_embedding = tsne.fit_transform(np.array(embeddings))
            elif mode_decomposition == 'umap':
                mapper = umap.UMAP(random_state=0)
                reduced_embedding = mapper.fit_transform(np.array(embeddings))

            pca_X = reduced_embedding.tolist()
            pca_Y = labels
            data = [x + [y1, y2, s[0], s[1]] for x, y1, y2, s in zip(pca_X, labels, eximplicits, sentences)]
            if mode_decomposition == 'pca':
                col1, col2 = 'PC1', 'PC2'
            elif mode_decomposition == 'tsne':
                col1, col2 = 'TSNE1', 'TSNE2'
            elif mode_decomposition == 'umap':
                col1, col2 = 'UMAP1', 'UMAP2'

            data = pd.DataFrame(data=data, columns=[col1, col2, 'category', 'eximplicit', 's1', 's2'])
            data = data[data['category'] != 'NONE']
            plt.figure()
            hue_order = ['CAUSE/REASON', 'CONCESSION', 'CONDITION', 'PURPOSE', 'JUSTIFICATION', 'CONTRAST']
            g = sns.scatterplot(data=data, x=col1, y=col2, hue='category', hue_order=hue_order, palette='Set1')

            # hue_order = ['Implicit', 'Explicit']
            # g = sns.scatterplot(data=data, x=col1, y=col2, hue='eximplicit', hue_order=hue_order, palette='Set1')

            # weights_cause = data[data['category'] == 'CAUSE/REASON'].mean().tolist()
            # weights_else = data[data['category'] == 'NONE'].mean().tolist()
            # plt.plot(weights_cause[0], weights_cause[1], marker='s', markersize=9, c='k')
            # plt.plot(weights_else[0], weights_else[1], marker='p', markersize=9, c='k')

            plt.legend(fontsize='xx-small', title='Category', title_fontsize='xx-small')
            plt.savefig(f'./test.scatterplot.{model_name.replace("/", ".")}.230806{METHOD}{TAG}.{mode_decomposition}.png')


def tensorboard_embedding(data_type='test'):
    rets = []
    num_seed = 1
    num_fold = 1
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接', 'その他根拠', '談話関係なし'])}
    # list_metrics = ['f1', 'precision', 'recall']
    METHOD_LIST ={'rand': ['.230413', '.rand'],
                'eos': ['.230413', '.eos'],
                'cls': ['.230301', '.cls'],
                'lconcat': ['.230227', '.concat'],
                'gconcat': ['.230418', '.gconcat'],
                'conj': ['.230507', '.conj']}
                # 'conj_init': ['.230311', '.conj.init']}
                # 'wonone_conj': ['.230430', '.wonone.conj'],
                # 'wonone_cls': ['.230502', '.wonone.cls']}
    MODE_LIST = ['xlm-roberta-large']

    with Path('./data/datasets.230201/disc_kwdlc/implicitlist.txt').open('r') as f:
        list_implicit_instances = f.readlines()
        list_implicit_instances = set([item.strip() for item in list_implicit_instances])

    with Path('./data/datasets.230201/disc_kwdlc/backwardlist.txt').open('r') as f:
        list_backward_instances = f.readlines()
        list_backward_instances = set([item.strip() for item in list_backward_instances])

    writer = SummaryWriter()
    methods = []
    for method, v in METHOD_LIST.items():
        print(method)
        methods.append(method)

        TAG = v[0]
        METHOD = v[1]

        for mode in MODE_LIST:
            sentence_pairs = {}
            golds = {}
            preds = {}
            info = {}
            last_hidden_states = {}
            model_name, OUTPUT_PATH, _ = get_properties(mode)
            OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
            for ifold in range(num_fold):
                for iseed in range(num_seed):
                    files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                    file = torch.load(files[0])
                    for items in file[f'{data_type}_results']:
                        key = f'{items["id"]}-{items["id_s1"]}-{items["id_s2"]}'

                        if key not in sentence_pairs:
                            sentence_pairs[key] = [items['s1'], items['s2']]
                        else:
                            pass

                        if key not in golds:
                            golds[key] = items['reason']
                        else:
                            pass

                        if key not in last_hidden_states:
                            if method == 'cls':
                                last_hidden_states[key] = [torch.as_tensor(items['last_hidden_state'][0]).tolist()]
                            else:
                                last_hidden_states[key] = [torch.as_tensor(items['last_hidden_state']).tolist()]
                        else:
                            if method == 'cls':
                                last_hidden_states[key].append(torch.as_tensor(items['last_hidden_state'][0]).tolist())
                            else:
                                last_hidden_states[key].append(torch.as_tensor(items['last_hidden_state']).tolist())

                        if key not in info:
                            info[key] = [f'{ifold}']
                        else:
                            if f'{ifold}' not in set(info[key]):
                                info[key].append(f'{ifold}')

            label_indexer = file['label_indexer']
            rev_label_indexer = {v: k for k, v in label_indexer.items()}
            for k in last_hidden_states.keys():
                if k not in preds:
                    preds[k] = [np.mean(last_hidden_states[k], axis=0).tolist()]
                else:
                    preds[k].append(np.mean(last_hidden_states[k], axis=0).tolist())


            dict_label = {'原因・理由': '原因・理由', '目的': '目的', '条件': '条件', '根拠': '根拠', '対比': '対比', '逆接': '逆接・譲歩', 'その他根拠': 'その他根拠', '談話関係なし': '談話関係なし'}
            dict_label_en = {'原因・理由': 'CAUSE/REASON', '目的': 'PURPOSE', '条件': 'CONDITION', '根拠': 'JUSTIFICATION', '対比': 'CONTRAST', '逆接・譲歩': 'CONCESSION', 'その他根拠': 'MISC', '談話関係なし': 'NONE'}

            embeddings = [v[0] for v in preds.values()]
            labels = ['Implicit, ' + dict_label_en[dict_label[golds[k]]] + ', ' + ', '.join(sentence_pairs[k]) if k in list_implicit_instances else 'Explicit, ' + dict_label_en[dict_label[golds[k]]] + ', ' + ', '.join(sentence_pairs[k]) for k in preds.keys()]
            sentences = [sentence_pairs[k] for k in preds.keys()]

            writer.add_embedding(mat=torch.as_tensor(embeddings, dtype=torch.float), metadata=labels, tag=method)
    writer.close()


if __name__ == '__main__':
    print('start')
    # rets = gather_details_by_categories(data_type='dev')
    # rets = gather_details_by_categories(data_type='test')
    # rets = gather_details_by_categories_avg_by_seed(data_type='dev')
    # rets = gather_details_by_categories_avg_by_seed(data_type='test')
    gather_embeddings(data_type='test')
    # gather_embeddings_by_cases(data_type='test')
    # tensorboard_embedding(data_type='test')
    print('end')
