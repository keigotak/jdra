# da.230310.0@ridel
import pandas as pd
from pathlib import Path
import re
import glob
import torch
from statistics import stdev
import decimal
decimal.getcontext().prec = 4
import json
from Helperfunctions import get_metrics_scores, get_metrics_scores_by_divided_type

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import dask.array as da
# from dask_ml.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
import umap

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

MODE_LIST = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
# MODE_LIST = ['tohoku-bert', 'xlm-roberta-large']
METHOD_LIST ={'rand': ['.230413', '.rand'],
              'eos': ['.230413', '.eos'],
              'cls': ['.230301', '.cls'],
              'lconcat': ['.230227', '.concat'], # local concatenate
              'gconcat': ['.230418', '.gconcat'],
              'conj': ['.230223', '.conj'],
              'wonone_conj': ['.230430', '.wonone.conj'],
              'wonone_cls': ['.230502', '.wonone.cls'],
              'conj_init': ['.230311', '.conj.init']}
METHOD = 'wonone_conj'
TAG = METHOD_LIST[METHOD][0]
METHOD = METHOD_LIST[METHOD][1]

def get_direction_annotations(resource='expert'):
    with Path(f'./data/datasets.230201/disc_{resource}.private.tsv').open('r') as f:
        texts = [l.strip().split('\t') for l in f.readlines()]

    all_datasets = {}
    for text in texts:
        numerics = re.findall(r'\d+', text[2])
        id_s1, id_s2 = int(numerics[0]) - 1, int(numerics[1]) - 1
        direction = re.findall(r'順方向|逆方向', text[4])
        if len(direction) > 1:
            all_datasets[f"{text[0]}-{id_s1}-{id_s2}"] = ['|'.join(direction), text[4]]
        elif len(direction) == 1:
            all_datasets[f"{text[0]}-{id_s1}-{id_s2}"] = [direction[0], text[4]]

    return all_datasets

def gather_overview():
    rets = []
    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG + f'/result{METHOD}.{model_name.replace("/", ".")}.csv'
        df = pd.read_csv(OUTPUT_PATH, header=0, index_col=None).sort_values('dev_semi_total_f1', ascending=False)
        df['model_name'] = [model_name for _ in range(df.shape[0])]
        df['output_path'] = [OUTPUT_PATH for _ in range(df.shape[0])]
        if len(rets) == 0:
            rets.append(df.columns.tolist())
        rets.extend(df[df['dev_semi_total_f1'] == df['dev_semi_total_f1'].max()].values.tolist())

    df = pd.DataFrame(rets[1:], columns=rets[0])
    df.to_csv(f'results{TAG}{METHOD}.csv', index=None)
    print(rets)

def gather_details():
    rets = []
    num_seed = 3
    num_fold = 5
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    label_indexer = {k: label_indexer[k] for k in ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']}
    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
        seed_results = {}
        dev_average_results, test_average_results = {item: [] for item in ['f1', 'precision', 'recall']}, {item: [] for item in ['f1', 'precision', 'recall']}
        for iseed in range(num_seed):
            dev_results, test_results = [], []
            for ifold in range(num_fold):
                files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                # data = torch.load(files[0])
                dev_results.extend(torch.load(files[0])['dev_results'])
                test_results.extend(torch.load(files[0])['test_results'])

            y_true = [r['label'] for r in dev_results]
            y_pred = [r['predicted_index'] for r in dev_results]
            dev_metrics = get_metrics_scores(y_true=y_true, y_pred=y_pred, label_indexer=label_indexer)

            y_true = [r['label'] for r in test_results]
            y_pred = [r['predicted_index'] for r in test_results]
            test_metrics = get_metrics_scores(y_true=y_true, y_pred=y_pred, label_indexer=label_indexer)

            for metrics in dev_average_results.keys():
                dev_average_results[metrics].append(dev_metrics['semi_total'][metrics])
                test_average_results[metrics].append(test_metrics['semi_total'][metrics])

            seed_results[iseed] = {'dev_results': dev_results.copy(), 'test_results': test_results.copy(), 'dev_metrics': dev_metrics, 'test_metrics': test_metrics}

        for metrics in dev_average_results.keys():
            seed_results[f'dev_avg_{metrics}'] = sum(dev_average_results[metrics]) / len(dev_average_results[metrics])
            seed_results[f'test_avg_{metrics}'] = sum(test_average_results[metrics]) / len(test_average_results[metrics])
            seed_results[f'dev_std_{metrics}'] = stdev(dev_average_results[metrics]) if len(dev_average_results[metrics]) > 1 else 0.0
            seed_results[f'test_std_{metrics}'] =  stdev(test_average_results[metrics]) if len(test_average_results[metrics]) > 1 else 0.0
        print(mode)
        dev_avg = ', '.join([str(seed_results[f'dev_avg_{k}']) for k in ['f1', 'precision', 'recall']])
        test_avg = ', '.join([str(seed_results[f'test_avg_{k}']) for k in ['f1', 'precision', 'recall']])
        dev_std = ', '.join([str(seed_results[f'dev_std_{k}']) for k in ['f1', 'precision', 'recall']])
        test_std = ', '.join([str(seed_results[f'test_std_{k}']) for k in ['f1', 'precision', 'recall']])
        print(f"dev: {dev_avg} | test: {test_avg}")
        rets.append(f"{mode},{dev_avg.replace(' ', '')},{dev_std.replace(' ', '')},{test_avg.replace(' ', '')},{test_std.replace(' ', '')},{OUTPUT_PATH}")

    with Path(f'./result{TAG}{METHOD}.csv').open('w') as f:
        f.write(','.join(['model'] + [f'dev_avg_{k}' for k in ['f1', 'precision', 'recall']] + [f'dev_std_{k}' for k in ['f1', 'precision', 'recall']] + [f'test_avg_{k}' for k in ['f1', 'precision', 'recall']] + [f'test_std_{k}' for k in ['f1', 'precision', 'recall']] + ['model_path']))
        f.write('\n')
        f.write('\n'.join(map(str, rets)))
        f.write('\n')

    print(rets)

def gather_details_by_categories():
    rets = []
    num_seed = 3
    num_fold = 5
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    label_indexer = {k: label_indexer[k] for k in ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']}
    list_metrics = ['f1', 'precision', 'recall']
    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
        seed_results = {}
        dev_average_results, test_average_results = {label: {item: [] for item in list_metrics} for label in label_indexer}, {label: {item: [] for item in list_metrics} for label in label_indexer}
        for iseed in range(num_seed):
            dev_results, test_results = [], []
            for ifold in range(num_fold):
                files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                # data = torch.load(files[0])
                dev_results.extend(torch.load(files[0])['dev_results'])
                test_results.extend(torch.load(files[0])['test_results'])

            y_true = [r['label'] for r in dev_results]
            y_pred = [r['predicted_index'] for r in dev_results]
            dev_metrics = get_metrics_scores(y_true=y_true, y_pred=y_pred, label_indexer=label_indexer)

            y_true = [r['label'] for r in test_results]
            y_pred = [r['predicted_index'] for r in test_results]
            test_metrics = get_metrics_scores(y_true=y_true, y_pred=y_pred, label_indexer=label_indexer)

            for metrics in list_metrics:
                for label in label_indexer:
                    dev_average_results[label][metrics].append(dev_metrics[label][metrics])
                    test_average_results[label][metrics].append(test_metrics[label][metrics])

            seed_results[iseed] = {'dev_results': dev_results.copy(), 'test_results': test_results.copy(), 'dev_metrics': dev_metrics, 'test_metrics': test_metrics}

        for metrics in list_metrics:
                for label in label_indexer:
                    seed_results[f'dev_avg_{label}_{metrics}'] = sum(dev_average_results[label][metrics]) / len(dev_average_results[label][metrics])
                    seed_results[f'test_avg_{label}_{metrics}'] = sum(test_average_results[label][metrics]) / len(test_average_results[label][metrics])
                    seed_results[f'dev_std_{label}_{metrics}'] = stdev(dev_average_results[label][metrics]) if len(dev_average_results[label][metrics]) > 1 else 0.0
                    seed_results[f'test_std_{label}_{metrics}'] =  stdev(test_average_results[label][metrics]) if len(test_average_results[label][metrics]) > 1 else 0.0
        print(mode)
        dev_avg = ', '.join([str(seed_results[f'dev_avg_{label}_{metrics}']) for metrics in list_metrics for label in label_indexer])
        test_avg = ', '.join([str(seed_results[f'test_avg_{label}_{metrics}']) for metrics in list_metrics for label in label_indexer])
        dev_std = ', '.join([str(seed_results[f'dev_std_{label}_{metrics}']) for metrics in list_metrics for label in label_indexer])
        test_std = ', '.join([str(seed_results[f'test_std_{label}_{metrics}']) for metrics in list_metrics for label in label_indexer])
        print(f"dev: {dev_avg} | test: {test_avg}")
        rets.append(f"{mode},{dev_avg.replace(' ', '')},{dev_std.replace(' ', '')},{test_avg.replace(' ', '')},{test_std.replace(' ', '')},{OUTPUT_PATH}")

    with Path(f'./result.bycategory{TAG}{METHOD}.csv').open('w') as f:
        f.write(','.join(['model'] + [f'dev_avg_{label}_{metrics}' for metrics in list_metrics for label in label_indexer] + [f'dev_std_{label}_{metrics}' for metrics in list_metrics for label in label_indexer] + [f'test_avg_{label}_{metrics}' for metrics in list_metrics for label in label_indexer] + [f'test_std_{label}_{metrics}' for metrics in list_metrics for label in label_indexer] + ['model_path']))
        f.write('\n')
        f.write('\n'.join(map(str, rets)))
        f.write('\n')

    print(rets)

def gather_details_by_direction():
    rets = []
    num_seed = 3
    num_fold = 5
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    label_indexer = {k: label_indexer[k] for k in ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']}
    list_metrics = ['f1', 'precision', 'recall']
    list_directions = ['forward', 'backward']

    detailed_labels = get_direction_annotations(resource='expert')

    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
        seed_results = {}
        dev_average_results, test_average_results = {d: {label: {item: [] for item in list_metrics} for label in label_indexer} for d in list_directions}, {d: {label: {item: [] for item in list_metrics} for label in label_indexer} for d in list_directions}
        for iseed in range(num_seed):
            dev_results, test_results = [], []
            for ifold in range(num_fold):
                files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                # data = torch.load(files[0])
                dev_results.extend(torch.load(files[0])['dev_results'])
                test_results.extend(torch.load(files[0])['test_results'])

            dev_results = [items | {'direction': detailed_labels[f"{items['id']}-{items['id_s1']}-{items['id_s2']}"][0], 'original_reason': detailed_labels[f"{items['id']}-{items['id_s1']}-{items['id_s2']}"][1]} for items in dev_results if f"{items['id']}-{items['id_s1']}-{items['id_s2']}" in detailed_labels.keys()]
            test_results = [items | {'direction': detailed_labels[f"{items['id']}-{items['id_s1']}-{items['id_s2']}"][0], 'original_reason': detailed_labels[f"{items['id']}-{items['id_s1']}-{items['id_s2']}"][1]} for items in test_results if f"{items['id']}-{items['id_s1']}-{items['id_s2']}" in detailed_labels.keys()]

            dev_metrics, test_metrics = {}, {}

            y_true_forward = [r['label'] for r in dev_results if r['direction'] == '順方向']
            y_pred_forward = [r['predicted_index'] for r in dev_results if r['direction'] == '順方向']
            dev_metrics['forward'] = get_metrics_scores_by_divided_type(y_true=y_true_forward, y_pred=y_pred_forward, label_indexer=label_indexer)

            y_true_backward = [r['label'] for r in dev_results if r['direction'] == '逆方向']
            y_pred_backward = [r['predicted_index'] for r in dev_results if r['direction'] == '逆方向']
            dev_metrics['backward'] = get_metrics_scores_by_divided_type(y_true=y_true_backward, y_pred=y_pred_backward, label_indexer=label_indexer)

            y_true_forward = [r['label'] for r in test_results if r['direction'] == '順方向']
            y_pred_forward = [r['predicted_index'] for r in test_results if r['direction'] == '順方向']
            test_metrics['forward'] = get_metrics_scores(y_true=y_true_forward, y_pred=y_pred_forward, label_indexer=label_indexer)

            y_true_backward = [r['label'] for r in test_results if r['direction'] == '逆方向']
            y_pred_backward = [r['predicted_index'] for r in test_results if r['direction'] == '逆方向']
            test_metrics['backward'] = get_metrics_scores(y_true=y_true_backward, y_pred=y_pred_backward, label_indexer=label_indexer)

            for metrics in list_metrics:
                for label in label_indexer:
                    for direction in list_directions:
                        dev_average_results[direction][label][metrics].append(dev_metrics[direction][label][metrics])
                        test_average_results[direction][label][metrics].append(test_metrics[direction][label][metrics])

            seed_results[iseed] = {'dev_results': dev_results.copy(), 'test_results': test_results.copy(), 'dev_metrics': dev_metrics, 'test_metrics': test_metrics}

        for metrics in list_metrics:
                for label in label_indexer:
                    for direction in list_directions:
                        seed_results[f'dev_avg_{label}_{direction}_{metrics}'] = sum(dev_average_results[direction][label][metrics]) / len(dev_average_results[direction][label][metrics])
                        seed_results[f'test_avg_{label}_{direction}_{metrics}'] = sum(test_average_results[direction][label][metrics]) / len(test_average_results[direction][label][metrics])
                        seed_results[f'dev_std_{label}_{direction}_{metrics}'] = stdev(dev_average_results[direction][label][metrics]) if len(dev_average_results[direction][label][metrics]) > 1 else 0.0
                        seed_results[f'test_std_{label}_{direction}_{metrics}'] =  stdev(test_average_results[direction][label][metrics]) if len(test_average_results[direction][label][metrics]) > 1 else 0.0
        print(mode)
        dev_avg = ', '.join([str(seed_results[f'dev_avg_{label}_{direction}_{metrics}']) for metrics in list_metrics for label in label_indexer for direction in list_directions])
        test_avg = ', '.join([str(seed_results[f'test_avg_{label}_{direction}_{metrics}']) for metrics in list_metrics for label in label_indexer for direction in list_directions])
        dev_std = ', '.join([str(seed_results[f'dev_std_{label}_{direction}_{metrics}']) for metrics in list_metrics for label in label_indexer for direction in list_directions])
        test_std = ', '.join([str(seed_results[f'test_std_{label}_{direction}_{metrics}']) for metrics in list_metrics for label in label_indexer for direction in list_directions])
        print(f"dev: {dev_avg} | test: {test_avg}")
        rets.append(f"{mode},{dev_avg.replace(' ', '')},{dev_std.replace(' ', '')},{test_avg.replace(' ', '')},{test_std.replace(' ', '')},{OUTPUT_PATH}")

    with Path(f'./result.bydirection{TAG}{METHOD}.csv').open('w') as f:
        f.write(','.join(['model'] + [f'dev_avg_{label}_{direction}_{metrics}' for metrics in list_metrics for label in label_indexer for direction in list_directions] + [f'dev_std_{label}_{direction}_{metrics}' for metrics in list_metrics for label in label_indexer for direction in list_directions] + [f'test_avg_{label}_{direction}_{metrics}' for metrics in list_metrics for label in label_indexer for direction in list_directions] + [f'test_std_{label}_{direction}_{metrics}' for metrics in list_metrics for label in label_indexer for direction in list_directions] + ['model_path']))
        f.write('\n')
        f.write('\n'.join(map(str, rets)))
        f.write('\n')

    print(rets)


def gather_embeddings():
    rets = []
    num_seed = 1
    num_fold = 1
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    label_indexer = {k: label_indexer[k] for k in ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']}
    dict_label = {'原因・理由': '原因・理由', '目的': '目的', '条件': '条件', '根拠': '根拠', '対比': '対比', '逆接': '逆接・譲歩', 'その他根拠': 'その他根拠', '談話関係なし': '談話関係なし'}
    dict_label_en = {'原因・理由': 'Cause or Reason', '目的': 'Purpose', '条件': 'Condition', '根拠': 'Justification', '対比': 'Contrast', '逆接・譲歩': 'Concession', 'その他根拠': 'その他根拠', '談話関係なし': '談話関係なし'}

    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
        seed_results = {}
        dev_average_results, test_average_results = {item: [] for item in ['f1', 'precision', 'recall']}, {item: [] for item in ['f1', 'precision', 'recall']}
        for iseed in range(num_seed):
            dev_results, test_results = [], []
            for ifold in range(num_fold):
                files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                # data = torch.load(files[0])
                dev_results.extend(torch.load(files[0])['dev_results'])
                test_results.extend(torch.load(files[0])['test_results'])
            
            dev_embeddings = {k: [] for k in label_indexer.keys()}
            for items in dev_results:
                dev_embeddings[dict_label[items['reason']]].append(items['last_hidden_state'] if 'cls' not in METHOD else items['last_hidden_state'][0])

            test_embeddings = {k: [] for k in label_indexer.keys()}
            for items in test_results:
                test_embeddings[dict_label[items['reason']]].append(items['last_hidden_state'] if 'cls' not in METHOD else items['last_hidden_state'][0])

            embeddings = test_embeddings
            results = test_results
            labels = ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠']
            # labels = ['原因・理由', '逆接・譲歩', '条件']
            labels = [k for k in labels if len(embeddings[k]) != 0]
            X = np.concatenate([embeddings[k] for k in labels if len(embeddings[k]) != 0])
            dX = X # da.from_array(X, chunks=X.shape)
            mode_decomposition = 'umap'
            if mode_decomposition == 'pca':
                pca = PCA(n_components=2)
                pca.fit(dX)
                pca_embedding = {k: pca.transform(embeddings[k]) for k in labels} # {k: pca.transform(embeddings[k]).compute() for k in labels}
            elif mode_decomposition == 'tsne':
                tsne = TSNE(n_components=2, random_state=0)
                tsne_embeddings = tsne.fit_transform(np.array([embedding for k in labels for embedding in embeddings[k]]))
                pca_Y = [label for label in labels for _ in range(len(embeddings[label]))]
                pca_embedding = {k: [] for k in labels}
                for k, e in zip(pca_Y, tsne_embeddings):
                    pca_embedding[k].append(e)
            elif mode_decomposition == 'umap':
                mapper = umap.UMAP(random_state=0)
                umap_embeddings = mapper.fit_transform(np.array([embedding for k in labels for embedding in embeddings[k]]))
                pca_Y = [label for label in labels for _ in range(len(embeddings[label]))]
                pca_embedding = {k: [] for k in labels}
                for k, e in zip(pca_Y, umap_embeddings):
                    pca_embedding[k].append(e)

            text_model = 'BERT'
            if 'gpt' in model_name:
                text_model = 'GPT-2'
            elif 't5' in model_name:
                text_model = 'T5'
            elif 'roberta' in model_name:
                text_model = 'RoBEERTa'

            pca_X = np.concatenate([pca_embedding[k] for k in labels]).tolist()
            pca_Y = [dict_label_en[label] for label in labels for _ in range(len(embeddings[label]))]
            data = [x + [y] for x, y in zip(pca_X, pca_Y)]
            data = pd.DataFrame(data=data, columns=['UMAP1', 'UMAP2', 'category'])
            plt.figure()
            g = sns.scatterplot(data=data, x='UMAP1', y='UMAP2', hue='category', palette='Set1')
            plt.savefig(f'./scatterplot.{model_name.replace("/", ".")}{METHOD}{TAG}.{mode_decomposition}.png')

def gather_confusion_matrics():
    rets = []
    num_seed = 1
    num_fold = 5
    key_labels = ['原因・理由', '目的', '条件', '根拠', '対比', '逆接', 'その他根拠', '談話関係なし']
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    dict_label = {'原因・理由': '原因・理由', '目的': '目的', '条件': '条件', '根拠': '根拠', '対比': '対比', '逆接': '逆接・譲歩', 'その他根拠': 'その他根拠', '談話関係なし': '談話関係なし'}
    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + METHOD + TAG
        seed_results = {}
        dev_average_results, test_average_results = {item: [] for item in ['f1', 'precision', 'recall']}, {item: [] for item in ['f1', 'precision', 'recall']}
        for iseed in range(num_seed):
            dev_results, test_results = [], []
            for ifold in range(num_fold):
                files = sorted(glob.glob(OUTPUT_PATH + f'.{iseed}' + f'/*fold{ifold}*.pt'), key=natural_keys)
                dev_results.append(torch.load(files[0])['dev_metrics'])
                test_results.append(torch.load(files[0])['test_metrics'])

            dev_metrics = {k: [results[k] for results in dev_results] for k in key_labels}
            test_metrics = {k: [results[k] for results in test_results] for k in key_labels}

            dev_embeddings = {k: [] for k in label_indexer.keys()}
            for items in dev_results:
                dev_embeddings[dict_label[items['reason']]].append(items['last_hidden_state'])

            test_embeddings = {k: [] for k in label_indexer.keys()}
            for items in test_results:
                test_embeddings[dict_label[items['reason']]].append(items['last_hidden_state'])



def compare_details():
    files = ['./result.230223.conj.csv', './result.230301.cls.csv', './result.230413.eos.csv', './result.230413.rand.csv', './result.230227.concat.csv', './result.230418.gconcat.csv', './result.230430.wonone.conj.csv', './result.230502.wonone.cls.csv']
    tag1 = ['CONJ', 'CLS', 'EOS', 'RAND', 'LCONC', 'GCONC', 'CONJ w/o no rel.', 'CLS w/o no rel.']
    tag2 = ['f1', 'precision', 'recall']
    tag3 = ['test', 'dev']
    models = ['tohoku-bert', 'nlp-waseda-roberta-base-japanese', 'rinna-roberta', 'nlp-waseda-roberta-large-japanese', 'xlm-roberta-base', 'xlm-roberta-large', 't5-base-encoder', 't5-base-decoder', 'rinna-japanese-gpt-1b', 'rinna-gpt2']
    model_tags = {'rinna-gpt2': 'GPT2', 'tohoku-bert': 'BERT', 't5-base-encoder': 'T5', 't5-base-decoder': 'T5', 'rinna-roberta': 'RoBERTa', 'nlp-waseda-roberta-base-japanese': 'RoBERTa', 'nlp-waseda-roberta-large-japanese': 'RoBERTa', 'rinna-japanese-gpt-1b': 'GPT', 'xlm-roberta-large': 'XLM', 'xlm-roberta-base': 'XLM'}
    header = ['', 'Test', 'Dev', 'Test', 'Dev', 'Test', 'Dev']
    rets = {t1: [] for t1 in tag1}
    for file, t1 in zip(files, tag1):
        df = pd.read_csv(file, index_col=None, header=0)
        df = df.T.to_dict()
        sorted_df = []
        for model in models:
            for items in df.values():
                if items['model'] == model:
                    sorted_df.append(items.copy())
        df = sorted_df
        for items in df:
            ret = [f"{t1} {model_tags[items['model']]}"]
            for t2 in tag2:
                for t3 in tag3:
                    avg = decimal.Decimal(str(items[f'{t3}_avg_{t2}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                    std = decimal.Decimal(str(items[f'{t3}_std_{t2}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                    ret.append(rf'{avg}$\pm{{{std}}}$')
            rets[t1].append(ret.copy())
    
    with Path('./result.table.230501.csv').open('w') as f:
        f.write(' & '.join(header) + ' \\\\\n')
        f.write(' & '.join(['BERT \cite{omura-kurohashi-2022-improving}', '47.0 $\pm{2.4}$', '-', '55.9 $\pm{1.1}$', '-', '41.0 $\pm{2.9}$', '-']) + ' \\\\\n')
        f.write(' & '.join(['XLM \cite{omura-kurohashi-2022-improving}', ' 51.9 $\pm{0.2}$', '-', '57.8 $\pm{2.3}$', '-', '48.2 $\pm{0.3}$', '-']) + ' \\\\\n')
        f.write('\\hline\n\\hline\n')
        for t1 in tag1:
            f.write('\n'.join([' & '.join(ret) + ' \\\\' for ret in rets[t1]]))
            f.write('\n\\hline\n\\hline\n')


def compare_details_by_categories():
    files = ['./result.bycategory.230223.conj.csv', './result.bycategory.230301.cls.csv', './result.bycategory.230413.eos.csv', './result.bycategory.230413.rand.csv', './result.bycategory.230227.concat.csv', './result.bycategory.230418.gconcat.csv', './result.bycategory.230430.wonone.conj.csv', './result.bycategory.230502.wonone.cls.csv']
    methods = ['CONJ', 'CLS', 'EOS', 'RAND', 'LCONC', 'GCONC', 'CONJ w/o no rel.', 'CLS w/o no rel.']
    metricses = ['f1', 'precision', 'recall']
    # tag2 = ['f1']
    modes = ['test', 'dev']
    key_labels = ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']
    # key_labels = ['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠']
    dict_label_en = {'原因・理由': 'Cause or Reason', '目的': 'Purpose', '条件': 'Condition', '根拠': 'Justification', '対比': 'Contrast', '逆接・譲歩': 'Concession', 'その他根拠': 'Misc', '談話関係なし': 'None'}
    # models = ['tohoku-bert', 'nlp-waseda-roberta-base-japanese', 'rinna-roberta', 'xlm-roberta-base', 'nlp-waseda-roberta-large-japanese', 'xlm-roberta-large', 't5-base-encoder', 't5-base-decoder', 'rinna-japanese-gpt-1b', 'rinna-gpt2']
    models = ['rinna-roberta']
    model_tags = {'rinna-gpt2': 'GPT2', 'tohoku-bert': 'BERT', 't5-base-encoder': 'T5', 't5-base-decoder': 'T5', 'rinna-roberta': 'RoBERTa', 'nlp-waseda-roberta-base-japanese': 'RoBERTa', 'nlp-waseda-roberta-large-japanese': 'RoBERTa', 'rinna-japanese-gpt-1b': 'GPT', 'xlm-roberta-large': 'XLM', 'xlm-roberta-base': 'XLM'}
    header = ['', 'Test', 'Dev', 'Test', 'Dev', 'Test', 'Dev']
    rets = {method: {} for method in methods}
    for file, method in zip(files, methods):
        df = pd.read_csv(file, index_col=None, header=0)
        df = df.T.to_dict()
        sorted_df = []
        for model in models:
            for items in df.values():
                if items['model'] == model:
                    sorted_df.append(items.copy())
        df = sorted_df
        for items in df:
            ret = {'method': f"{method}", 'model': items['model']}
            for metrics in metricses:
                for label in key_labels:
                    for mode in modes:
                        avg = decimal.Decimal(str(items[f'{mode}_avg_{label}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                        std = decimal.Decimal(str(items[f'{mode}_std_{label}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                        ret[f'{mode}_{label}_{metrics}'] = f'{avg}$\pm{{{std}}}$'
            rets[method][items['model']] = ret.copy()
    
    with Path('./result.table.bycategories.230506.csv').open('w') as f:
        f.write('|'.join(models) + '\n')
        for i, label in enumerate(key_labels):
            flg_label_en = True
            flg_label_jp = True
            # f.write('\hline\n\multicolumn{7}{l}{' + f'{label} {dict_label_en[label]}' + '}\\\\\n')
            # f.write(' & '.join(['BERT \cite{omura-kurohashi-2022-improving}', '47.0 $\pm{2.4}$', '-', '55.9 $\pm{1.1}$', '-', '41.0 $\pm{2.9}$', '-']) + ' \\\\\n')
            # f.write(' & '.join(['XLM \cite{omura-kurohashi-2022-improving}', ' 51.9 $\pm{0.2}$', '-', '57.8 $\pm{2.3}$', '-', '48.2 $\pm{0.3}$', '-']) + ' \\\\\n')
            for j, method in enumerate(methods):
                for model in models:
                    tmp = [method]
                    items = [method]
                    for metrics in metricses:
                        for mode in modes:
                            tmp.append(f'{mode}_{label}_{metrics}')
                            items.append(rets[method][model][f'{mode}_{label}_{metrics}'])

                    if flg_label_en:
                        f.write('\n'.join([' & '.join([dict_label_en[label]] + items)]) + ' \\\\\n')
                        flg_label_en = False
                    else:
                        if flg_label_jp:
                            f.write('\n'.join([' & '.join([label] + items)]) + ' \\\\\n')
                            flg_label_jp = False
                        else:
                            f.write('\n'.join([' & '.join([''] + items)]) + ' \\\\\n')
            f.write('\hline\n')

def compare_details_by_categories_wo_no_relations():
    files = ['./result.bycategory.230430.wonone.conj.csv', './result.bycategory.230502.wonone.cls.csv']
    methods = ['CONJ', 'CLS']
    metricses = ['f1', 'precision', 'recall']
    modes = ['test', 'dev']
    key_labels = ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']
    dict_label_en = {'原因・理由': 'Cause or Reason', '目的': 'Purpose', '条件': 'Condition', '根拠': 'Justification', '対比': 'Contrast', '逆接・譲歩': 'Concession', 'その他根拠': 'Misc', '談話関係なし': 'None'}
    # models = ['tohoku-bert', 'nlp-waseda-roberta-base-japanese', 'rinna-roberta', 'xlm-roberta-base', 'nlp-waseda-roberta-large-japanese', 'xlm-roberta-large', 't5-base-encoder', 't5-base-decoder', 'rinna-japanese-gpt-1b', 'rinna-gpt2']
    models = ['xlm-roberta-large', 'rinna-roberta', 'tohoku-bert']
    model_tags = {'rinna-gpt2': 'GPT2', 'tohoku-bert': 'BERT\\footnotemark[1]', 't5-base-encoder': 'T5', 't5-base-decoder': 'T5', 'rinna-roberta': 'RoBERTa\\footnotemark[3]', 'nlp-waseda-roberta-base-japanese': 'RoBERTa', 'nlp-waseda-roberta-large-japanese': 'RoBERTa', 'rinna-japanese-gpt-1b': 'GPT', 'xlm-roberta-large': 'XLM\\footnotemark[6]', 'xlm-roberta-base': 'XLM'}
    header = ['', 'Test', 'Dev', 'Test', 'Dev', 'Test', 'Dev']
    rets = {method: {} for method in methods}
    for file, method in zip(files, methods):
        df = pd.read_csv(file, index_col=None, header=0)
        df = df.T.to_dict()
        sorted_df = []
        for model in models:
            for items in df.values():
                if items['model'] == model:
                    sorted_df.append(items.copy())
        df = sorted_df
        for items in df:
            ret = {'method': f"{method}", 'model': items['model']}
            for metrics in metricses:
                for label in key_labels:
                    for mode in modes:
                        avg = decimal.Decimal(str(items[f'{mode}_avg_{label}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                        std = decimal.Decimal(str(items[f'{mode}_std_{label}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                        ret[f'{mode}_{label}_{metrics}'] = f'{avg}$\pm{{{std}}}$'
            rets[method][items['model']] = ret.copy()
    
    with Path('./result.table.bycategories.wonorelation.230506.csv').open('w') as f:
        f.write('|'.join(models) + '\n')
        for i, label in enumerate(key_labels):
            flg_label_en = True
            # f.write('\hline\n\multicolumn{7}{l}{' + f'{label} {dict_label_en[label]}' + '}\\\\\n')
            # f.write(' & '.join(['BERT \cite{omura-kurohashi-2022-improving}', '47.0 $\pm{2.4}$', '-', '55.9 $\pm{1.1}$', '-', '41.0 $\pm{2.9}$', '-']) + ' \\\\\n')
            # f.write(' & '.join(['XLM \cite{omura-kurohashi-2022-improving}', ' 51.9 $\pm{0.2}$', '-', '57.8 $\pm{2.3}$', '-', '48.2 $\pm{0.3}$', '-']) + ' \\\\\n')
            for j, method in enumerate(methods):
                flg_method = True
                for model in models:
                    tmp = []
                    items = []
                    for metrics in metricses:
                        for mode in modes:
                            tmp.append(f'{mode}_{label}_{metrics}')
                            items.append(rets[method][model][f'{mode}_{label}_{metrics}'])

                    if flg_label_en:
                        f.write('\n'.join([' & '.join([dict_label_en[label], method, model_tags[model]] + items)]) + ' \\\\\n')
                        flg_label_en = False
                        flg_method = False
                    else:
                        if flg_method:
                            f.write('\n'.join([' & '.join(['', method, model_tags[model]] + items)]) + ' \\\\\n')
                            flg_method = False
                        else:
                            f.write('\n'.join([' & '.join(['', '', model_tags[model]] + items)]) + ' \\\\\n')
            f.write('\hline\n')

def compare_details_by_directions_xlm_bymethods():
    files = ['./result.bydirection.230223.conj.csv', './result.bydirection.230301.cls.csv', './result.bydirection.230413.eos.csv']
    methods = ['CONJ', 'CLS', 'EOS']
    metricses = ['f1', 'precision', 'recall']
    list_directions = ['forward', 'backward']
    dict_directions = {'forward': 'Forward', 'backward': 'Backward'}
    modes = ['test', 'dev']
    key_labels = ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']
    # key_labels = ['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠']
    dict_label_en = {'原因・理由': 'Cause or Reason', '目的': 'Purpose', '条件': 'Condition', '根拠': 'Justification', '対比': 'Contrast', '逆接・譲歩': 'Concession', 'その他根拠': 'Misc', '談話関係なし': 'None'}

    # models = ['tohoku-bert', 'nlp-waseda-roberta-base-japanese', 'rinna-roberta', 'xlm-roberta-base', 'nlp-waseda-roberta-large-japanese', 'xlm-roberta-large', 't5-base-encoder', 't5-base-decoder', 'rinna-japanese-gpt-1b', 'rinna-gpt2']
    # models = ['tohoku-bert', 'xlm-roberta-large']
    models = ['xlm-roberta-large']
    model_tags = {'rinna-gpt2': 'GPT2', 'tohoku-bert': 'BERT', 't5-base-encoder': 'T5', 't5-base-decoder': 'T5', 'rinna-roberta': 'RoBERTa', 'nlp-waseda-roberta-base-japanese': 'RoBERTa', 'nlp-waseda-roberta-large-japanese': 'RoBERTa', 'rinna-japanese-gpt-1b': 'GPT', 'xlm-roberta-large': 'XLM', 'xlm-roberta-base': 'XLM'}
    header = ['', 'Test', 'Dev', 'Test', 'Dev', 'Test', 'Dev']
    rets = {method: {} for method in methods}
    for file, method in zip(files, methods):
        df = pd.read_csv(file, index_col=None, header=0)
        df = df.T.to_dict()
        sorted_df = []
        for model in models:
            for items in df.values():
                if items['model'] == model:
                    sorted_df.append(items.copy())

        df = sorted_df
        for items in df:
            ret = {'method': f"{method}", 'model': items['model']}
            for metrics in metricses:
                for direction in list_directions:
                    for label in key_labels:
                        for mode in modes:
                            avg = decimal.Decimal(str(items[f'{mode}_avg_{label}_{direction}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                            std = decimal.Decimal(str(items[f'{mode}_std_{label}_{direction}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                            ret[f'{mode}_{label}_{direction}_{metrics}'] = f'{avg}$\pm{{{std}}}$'
            rets[method][items['model']] = ret.copy()
    
    
    with Path('./result.table.bydirections.xlm.bymethods.230504.csv').open('w') as f:
        for j, direction in enumerate(list_directions):
            f.write('\hline\n')
            f.write(dict_directions[direction] + ' \\\\\n')
            f.write('\hline\n')
            for i, label in enumerate(key_labels):
                flg_label = True
                for method in methods:
                    for model in rets[method].keys():
                        tmp = []
                        items = [direction, model_tags[model]]
                        for metrics in metricses:
                            for mode in modes:
                                tmp.append(f'{mode}_{label}_{direction}_{metrics}')
                                items.append(rets[method][model][f'{mode}_{label}_{direction}_{metrics}'])
                        if flg_label:
                            f.write('\n'.join([' & '.join([dict_label_en[label], method] + items[2:])]) + ' \\\\\n')
                            flg_label = False
                        else:
                            f.write('\n'.join([' & '.join(['', method] + items[2:])]) + ' \\\\\n')
                f.write('\hline\n')

def compare_details_by_directions_conj_bymodels():
    files = ['./result.bydirection.230223.conj.csv']
    methods = ['CONJ']
    metricses = ['f1', 'precision', 'recall']
    list_directions = ['forward', 'backward']
    dict_directions = {'forward': 'Forward', 'backward': 'Backward'}
    modes = ['test', 'dev']
    key_labels = ['原因・理由', '逆接・譲歩', '条件', '目的', '対比', '根拠', 'その他根拠', '談話関係なし']
    # key_labels = ['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠']
    dict_label_en = {'原因・理由': 'Cause or Reason', '目的': 'Purpose', '条件': 'Condition', '根拠': 'Justification', '対比': 'Contrast', '逆接・譲歩': 'Concession', 'その他根拠': 'Misc', '談話関係なし': 'None'}

    # models = ['tohoku-bert', 'nlp-waseda-roberta-base-japanese', 'rinna-roberta', 'xlm-roberta-base', 'nlp-waseda-roberta-large-japanese', 'xlm-roberta-large', 't5-base-encoder', 't5-base-decoder', 'rinna-japanese-gpt-1b', 'rinna-gpt2']
    models = ['xlm-roberta-large', 'rinna-roberta', 'tohoku-bert']
    model_tags = {'rinna-gpt2': 'GPT2', 'tohoku-bert': 'BERT\\footnotemark[1]', 't5-base-encoder': 'T5', 't5-base-decoder': 'T5', 'rinna-roberta': 'RoBERTa\\footnotemark[3]', 'nlp-waseda-roberta-base-japanese': 'RoBERTa', 'nlp-waseda-roberta-large-japanese': 'RoBERTa', 'rinna-japanese-gpt-1b': 'GPT', 'xlm-roberta-large': 'XLM\\footnotemark[6]', 'xlm-roberta-base': 'XLM'}
    header = ['', 'Test', 'Dev', 'Test', 'Dev', 'Test', 'Dev']
    rets = {method: {} for method in methods}
    for file, method in zip(files, methods):
        df = pd.read_csv(file, index_col=None, header=0)
        df = df.T.to_dict()
        sorted_df = []
        for model in models:
            for items in df.values():
                if items['model'] == model:
                    sorted_df.append(items.copy())

        df = sorted_df
        for items in df:
            ret = {'method': f"{method}", 'model': items['model']}
            for metrics in metricses:
                for direction in list_directions:
                    for label in key_labels:
                        for mode in modes:
                            avg = decimal.Decimal(str(items[f'{mode}_avg_{label}_{direction}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                            std = decimal.Decimal(str(items[f'{mode}_std_{label}_{direction}_{metrics}'] * 100)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP)
                            ret[f'{mode}_{label}_{direction}_{metrics}'] = f'{avg}$\pm{{{std}}}$'
            rets[method][items['model']] = ret.copy()
    
    
    with Path('./result.table.bydirections.conj.bymodels.230504.csv').open('w') as f:
        for j, direction in enumerate(list_directions):
            f.write('\hline\n')
            f.write(dict_directions[direction] + ' \\\\\n')
            f.write('\hline\n')
            for i, label in enumerate(key_labels):
                flg_label = True
                for method in methods:
                    for model in rets[method].keys():
                        tmp = []
                        items = [direction, model_tags[model]]
                        for metrics in metricses:
                            for mode in modes:
                                tmp.append(f'{mode}_{label}_{direction}_{metrics}')
                                items.append(rets[method][model][f'{mode}_{label}_{direction}_{metrics}'])
                        if flg_label:
                            f.write('\n'.join([' & '.join([dict_label_en[label]] + items[1:])]) + ' \\\\\n')
                            flg_label = False
                        else:
                            f.write('\n'.join([' & '.join([''] + items[1:])]) + ' \\\\\n')

                f.write('\hline\n')


if __name__=='__main__':
    # gather_overview()
    gather_details() # 結果ファイル生成
    gather_embeddings() # scatter plot 生成
    gather_details_by_categories() # 分類別の結果ファイル生成
    gather_details_by_direction() # 方向別の分析
    compare_details() # 全体の結果集計後 tex 生成
    compare_details_by_categories() # 分類別に結果集計後 tex 生成
    compare_details_by_categories_wo_no_relations() # 分類別に結果集計後 tex 生成
    compare_details_by_directions_xlm_bymethods() # 方向別に結果集計後 tex 生成
    compare_details_by_directions_conj_bymodels() # 方向別に結果集計後 tex 生成
