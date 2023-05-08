from pathlib import Path
import re

import torch
from rhoknp import KWJA, Document


class Dataset(torch.utils.data.Dataset):
    def __init__(self, s1, s2, y, informations):
        super().__init__()
        self.s1 = s1
        self.s2 = s2
        self.informations = informations
        self.y = y
        self.len = len(y)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.s1[index], self.s2[index], self.y[index], self.informations[index]


def get_data(resource='expert'):
    modes = ['train', 'test']

    ids, texts = {}, {}
    for mode in modes:
        with Path(f'./data/KWDLC/id/{mode}.id').open('r') as f:
            texts = f.readlines()
        texts = [text.strip() for text in texts]
        ids[mode] = texts.copy()
    
    with Path(f'./data/KWDLC/disc/disc_{resource}.txt').open('r') as f:
        texts = f.readlines()
    
    problems = []
    sentences, discrosures = [], []
    for text in texts:
        text = text.strip()
        if text == '':
            problems.extend(discrosures.copy())
            sentences, discrosures = [], []
            continue
        elif text.startswith('# A-ID:'):
            problem_index = text.replace('# A-ID:', '')
        else:
            pattern = re.compile(r"[0-9]+-[0-9]+")
            if pattern.match(text):
                items = text.split(' ')
                numbers, original_reason = items[0].split('-'), items[1]
                sentence_index_from = int(numbers[0])
                sentence_index_to = int(numbers[1])

                if resource == 'crowdsourcing':
                    reason = original_reason.split(' ')[0].split(':')[0]
                    if reason == '逆接':
                        reason = '逆接・譲歩'
                elif resource == 'expert':
                    reason = re.sub(r"\(順方向\)|\(逆方向\)|\(方向なし\)", '', original_reason)
                discrosures.append({'id': sentences[sentence_index_from - 1][0], 's1': sentences[sentence_index_from - 1][2], 's2': sentences[sentence_index_to - 1][2], 'id_s1': sentence_index_from, 'id_s2': sentence_index_to, 'reason': reason, 'original_reason': original_reason})
            else:
                items = text.split(' ')
                sentences.append([problem_index, items[0], items[1]])    

    all_datasets = {}
    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    for mode in modes:
        datasets = []
        _ids = set(ids[mode])
        for p in problems:
            if p['id'] in _ids:
                datasets.append(p | {'mode': mode})
            else:
                pass
        all_datasets[mode] = datasets
    
    for mode in modes:
        for i in range(len(all_datasets[mode])):
            all_datasets[mode][i]['label'] = label_indexer[all_datasets[mode][i]['reason']]

    return all_datasets, label_indexer

def get_data_kishimoto():
    modes = {'crowdsourcing': 'train', 'expert': 'test'}
    resources = ['expert', 'crowdsourcing']
    
    problems = []
    sentences, discrosures = [], []

    all_datasets = {}
    for resource in resources:
        with Path(f'./data/KWDLC/disc/disc_{resource}.txt').open('r') as f:
            texts = f.readlines()
        
        for text in texts:
            text = text.strip()
            if text == '':
                problems.extend(discrosures.copy())
                sentences, discrosures = [], []
                continue
            elif text.startswith('# A-ID:'):
                problem_index = text.replace('# A-ID:', '')
            else:
                pattern = re.compile(r"[0-9]+-[0-9]+")
                if pattern.match(text):
                    items = text.split(' ')
                    numbers, original_reason = items[0].split('-'), items[1]
                    sentence_index_from = int(numbers[0])
                    sentence_index_to = int(numbers[1])

                    if resource == 'crowdsourcing':
                        reason = original_reason.split(' ')[0].split(':')[0]
                        if reason == '逆接':
                            reason = '逆接・譲歩'
                    elif resource == 'expert':
                        reason = re.sub(r"\(順方向\)|\(逆方向\)|\(方向なし\)", '', original_reason)
                    discrosures.append({'id': sentences[sentence_index_from - 1][0], 's1': sentences[sentence_index_from - 1][2], 's2': sentences[sentence_index_to - 1][2], 'id_s1': sentence_index_from, 'id_s2': sentence_index_to, 'reason': reason, 'original_reason': original_reason, 'source': resource})
                else:
                    items = text.split(' ')
                    sentences.append([problem_index, items[0], items[1]])

        all_datasets[resource] = problems.copy()

    label_indexer = {k: i for i, k in enumerate(['原因・理由', '目的', '条件', '根拠', '対比', '逆接・譲歩', 'その他根拠', '談話関係なし'])}
    for resource in resources:
        datasets = []
        for p in all_datasets[resource]:
            datasets.append(p | {'mode': modes[resource]})
        all_datasets[modes[resource]] = datasets
    
    for resource in resources:
        for i in range(len(all_datasets[modes[resource]])):
            all_datasets[modes[resource]][i]['label'] = label_indexer[all_datasets[modes[resource]][i]['reason']]

    return all_datasets, label_indexer

def split_to_dev(data):
    train_size = int(0.8 * len(data))
    return data[:train_size], data[train_size:]


def eval():
    INFERENCE_BATCH_SIZE = 32
    DATASET_MODE = 'expert' # crowdsourcing, expert, kishimoto


    if DATASET_MODE in set(['expert', 'crowdsourcing']):
        datasets, label_indexer = get_data(resource=DATASET_MODE) # expert, crowdsourcing
    elif DATASET_MODE in set(['kishimoto']):
        datasets, label_indexer = get_data_kishimoto()

    s1, s2, y = [d['s1'] for d in datasets['test']], [d['s2'] for d in datasets['test']], [d['label'] for d in datasets['test']]
    test_dataset = Dataset(s1, s2, y, datasets['test'])

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    kwja = KWJA()


    for sentence in test_dataloader:
        sent = kwja.apply_to_sentence(sentence)

        # Get information.
        if sent.need_clause_tag is True:
            print("KNP might be too old; please update it.")

        discourse_relations = []
        for clause in sent.clauses:
            discourse_relations.extend(clause.discourse_relations)

        if discourse_relations:
            print(f"Found {len(discourse_relations)} discourse relations:")
            for i, discourse_relation in enumerate(discourse_relations, start=1):
                modifier = discourse_relation.modifier
                head = discourse_relation.head
                label = discourse_relation.label
                print(f'  {i}. "{modifier}" -({label.value})-> "{head}"')
        else:
            print("No discourse relation found.")

if __name__=='__main__':
    eval()