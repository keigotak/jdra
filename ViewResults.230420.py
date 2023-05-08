from pathlib import Path
import pandas as pd
import glob
import torch
import re
import json


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

def gather_details_by_categories():
    output_file = 'total_list.230501.0.csv'

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
                'conj': ['.230223', '.conj'],
                'wonone_conj': ['.230430', '.wonone.conj'],
                'wonone_cls': ['.230502', '.wonone.cls'],
                'conj_init': ['.230311', '.conj.init']}
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
                    for items in torch.load(files[0])['test_results']:
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

if __name__ == '__main__':
    rets = gather_details_by_categories()
