from pathlib import Path
import json

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


if __name__=='__main__':
    all_data = []
    for i in range(5):
        all_data.append(get_data_230205(resource='expert', index_fold=i))

    print(all_data)