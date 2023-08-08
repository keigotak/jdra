import glob
import os
import pathlib
import tarfile
import neologdn
import pandas as pd

from pathlib import Path
import json

import scattertext as st
from scattertext import SampleCorpora, PhraseMachinePhrases, dense_rank, RankDifference, AssociationCompactor, produce_scattertext_explorer

import spacy
from collections import Counter
from itertools import chain
from IPython.display import HTML
import joblib

def get_data_230205(resource='expert', index_fold=0):
    modes = ['train', 'dev', 'test']

    all_datasets = {mode: [] for mode in modes}
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接', 'その他根拠', '談話関係なし'])}
    for mode in modes:
        with Path(f'./data/datasets.230201/disc_kwdlc/{resource}/fold_{index_fold+1}/{mode}.jsonl').open('r') as f:
            texts = [json.loads(l) for l in f.readlines()]

        for text in texts:
            all_datasets[mode].append({'id': text['document_id'],
            's1': text['sentences'][text['arg1_sent_index']].replace('【', '').replace('】', ''),
            's2': text['sentences'][text['arg2_sent_index']].replace('【', '').replace('】', ''),
            'seg_s1': text['segmented_sentences'][text['arg1_sent_index']].replace(' ', ''),
            'seg_s2': text['segmented_sentences'][text['arg2_sent_index']].replace(' ', ''),
            'id_s1': text['arg1_sent_index'],
            'id_s2': text['arg2_sent_index'],
            'reason': text['sense'],
            'annotator': text['annotator'],
            'label': label_indexer[text['sense']],
            'mode': mode})
            s1 = text['sentences'][text['arg1_sent_index']].replace('【', '').replace('】', '')
            s2 = text['sentences'][text['arg2_sent_index']].replace('【', '').replace('】', '')
            seg_s1 = text['segmented_sentences'][text['arg1_sent_index']].replace(' ', '')
            seg_s2 = text['segmented_sentences'][text['arg2_sent_index']].replace(' ', '')
            if s1 != seg_s1:
                print(f'{s1}\n{seg_s1}')
            if s2 != seg_s2:
                print(f'{s2}\n{seg_s2}')

    return all_datasets, label_indexer

# 品詞を絞りこみつつ、unigramの出現回数を集計
class UnigramSelectedPos(st.FeatsFromSpacyDoc):
    """
    品詞の絞り込みを行い、unigramをカウント
    デフォルトの絞り込み品詞は[固有名詞、名詞、動詞、形容詞、副詞]
    """
    def __init__(self):
        super().__init__()
        # self._use_pos = ['PROPN', 'NOUN', 'VERB', 'ADJ', 'ADV']
        # self._use_pos = ['ADP', 'VERB', 'PUNCT', 'NOUN', 'NUM', 'SYM', 'ADV', 'ADJ']
        self._use_pos = ['PUNCT']

    def get_feats(self, doc):
        # return Counter([c.lemma_ for c in doc])
        # return Counter([c.lemma_ for c in doc if c.pos_ in self._use_pos])
        return Counter([c.lemma_ for c in doc if c.pos_ not in self._use_pos])
        # def func(self, c):
        #     if c.pos_ in self._use_pos:
        #         return c.lemma_
        #     else:
        #         return None
        # rets = joblib.Parallel(n_jobs=-1)(joblib.delayed(func)(self, c) for c in doc)
        # return Counter(rets)

def evaluate_by_labels():
    all_data = []
    for i in range(5):
        all_data.append(get_data_230205(resource='crowd', index_fold=i))

    datasets = {}
    data_types = ['train', 'dev', 'test']
    # labels = ['談話関係なし']
    # labels = ['逆接']
    # labels = ['原因・理由', '逆接', '条件', '目的', '対比', '根拠']
    labels = ['原因・理由', '逆接', '条件', '目的', '対比', '根拠', '談話関係なし']
    for data_type in data_types:
        datasets[data_type] = [item for items in all_data for item in items[0][data_type]]

    mode_x = 'train'
    for mode in ['test']:
        for label in labels:
            print(label)
            data = [[d['s1'] + d['s2'], data_type] for data_type in data_types for d in datasets[data_type] if d['reason'] == label]
            df = pd.DataFrame(data=data, columns=["text", "category"])

            # Corpusの作成
            corpus = (st.CorpusFromPandas(df, 
                                        category_col='category', 
                                        text_col='text',
                                        nlp = spacy.load("ja_ginza"),
                                        feats_from_spacy_doc=UnigramSelectedPos()
                                        )
                    .build())
            html = st.produce_scattertext_explorer(
                    corpus,
                    category=mode, # y軸カテゴリ
                    not_categories=[mode_x], # x軸カテゴリ（複数選択可）
                    category_name=mode, # y軸ラベル
                    not_category_name=mode_x, # x軸ラベル
                    asian_mode=True, # 日本語モード
                    # minimum_term_frequency=0, # 指定より出現回数の多い単語のみをプロット
                    # max_terms=4000, # プロットする最大数
                    # pmi_threshold_coefficient=0,
                    width_in_pixels=1000,
                    transform=st.Scalers.dense_rank
                    # use_non_text_features=True,
                    # term_scorer=st.RankDifference()
                    # sort_by_dist=False
                    # topic_model_term_lists={term: [term] for term in corpus.get_metadata()},
                    # topic_model_preview_size=0, 
                    # use_full_doc=True
                    )

            # 散布図の表示
            open(f'./scattertext.all.{mode_x}{mode}.{label}.html', 'w').write(html)

def evaluage_together():
    all_data = []
    for i in range(5):
        all_data.append(get_data_230205(resource='crowd', index_fold=i))

    datasets = {}
    data_types = ['train', 'dev', 'test']
    labels = ['原因・理由', '逆接', '条件', '目的', '対比', '根拠'] # , '談話関係なし'
    for data_type in data_types:
        datasets[data_type] = [item for items in all_data for item in items[0][data_type]]

    for mode in ['test']:
        data = [[d['s1'] + d['s2'], data_type] for data_type in data_types for d in datasets[data_type]]
        df = pd.DataFrame(data=data, columns=["text", "category"])

        # Corpusの作成
        corpus = (st.CorpusFromPandas(df, 
                                    category_col='category', 
                                    text_col='text',
                                    nlp = spacy.load("ja_ginza"),
                                    feats_from_spacy_doc=UnigramSelectedPos()
                                    )
                .build())

        html = st.produce_scattertext_explorer(
                corpus,
                category=mode, # y軸カテゴリ
                not_categories=[mode], # x軸カテゴリ（複数選択可）
                category_name=mode, # y軸ラベル
                not_category_name=mode, # x軸ラベル
                asian_mode=True, # 日本語モード
                # minimum_term_frequency=0, # 指定より出現回数の多い単語のみをプロット
                # max_terms=4000, # プロットする最大数
                # pmi_threshold_coefficient=0,
                width_in_pixels=1000,
                transform=st.Scalers.dense_rank
                # use_non_text_features=True,
                # term_scorer=st.RankDifference()
                # sort_by_dist=False
                # topic_model_term_lists={term: [term] for term in corpus.get_metadata()},
                # topic_model_preview_size=0, 
                # use_full_doc=True
                )

        # 散布図の表示
        open(f'./scattertext.all.{mode}.html', 'w').write(html)

if __name__ == '__main__':
    print('start')
    evaluate_by_labels()
    evaluage_together()
    print('done')