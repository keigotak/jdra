import re
import pandas as pd
import datetime

import os
import random

import numpy as np
import torch
import torch.nn as nn

import json
from pathlib import Path
import transformers
transformers.BertTokenizer = transformers.BertJapaneseTokenizer

from accelerate import Accelerator
from transformers import T5Tokenizer, T5Model
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup
from dadaptation import DAdaptAdam

seed = 2
random.seed(seed)
torch.manual_seed(seed)
transformers.trainer_utils.set_seed(seed)


from BertJapaneseTokenizerFast import BertJapaneseTokenizerFast
from ValueWatcher import ValueWatcher
from Helperfunctions import get_metrics_scores

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_class):
        super(Classifier, self).__init__()
        # self.encoder_f = nn.ModuleList([nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)])
        # self.encoder_b = nn.ModuleList([nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)])
        # self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_class)
        # self.num_layers = num_layers
        # self.pooling_method = pooling_method
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        # for i in range(self.num_layers):
        #     for enc in [self.encoder_f[i], self.encoder_b[i]]:
        #         encoder_outputs, hidden = enc(x)
        #         x = encoder_outputs + x
        #         x = x[torch.arange(x.shape[0]-1, -1, -1), :, :]
        #         x = self.dropout(x)

        # x, weights = self.attention(x, x, x, key_padding_mask=mask==0)
        x = self.dropout(x)
        x = self.linear(x)

        # if self.pooling_method == 'mean':
        #     x = self.pooling_mean(x, mask)
        # elif self.pooling_method == 'max':
        #     x = self.pooling_max(x, mask)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, s1, y, informations):
        super().__init__()
        self.s1 = s1
        self.informations = informations
        self.y = y
        self.len = len(y)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.s1[index], self.y[index], self.informations[index]


def get_data(resource='expert', index_fold=0):
    modes = ['train', 'dev', 'test']

    ids, texts = {}, {}
    for mode in modes:
        with Path(f'./data/KWDLC-discourse-split/disc/split/cv/fold_{index_fold+1}/{mode}.{resource}.txt').open('r') as f:
            texts = f.readlines()
        texts = [text.strip() for text in texts]
        ids[mode] = texts.copy()
    
    with Path(f'./data/KWDLC-discourse-split/disc/disc_{resource}.txt').open('r') as f:
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

def get_data_kishimoto(dataset_mode='only_expert', index_fold=0):
    resources = ['expert'] if dataset_mode == 'only_expert' else ['expert', 'crowdsourcing']
    
    all_datasets = {}

    problems = []
    sentences, discrosures = [], []
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
    document_ids = []
    for document_id in [items['id'] for items in all_datasets['expert']]:
        if document_id not in document_ids:
            document_ids.append(document_id)
    train_document_ids, dev_document_ids, test_document_ids = split_to_dev_fold(document_ids, num_fold=5, index_fold=index_fold)

    all_datasets['train'] = [items | {'mode': 'train', 'label': label_indexer[items['reason']], 'source': 'expert'} for items in all_datasets['expert'] if items['id'] in set(train_document_ids)]
    all_datasets['dev'] = [items | {'mode': 'dev', 'label': label_indexer[items['reason']], 'source': 'expert'} for items in all_datasets['expert'] if items['id'] in set(dev_document_ids)]
    all_datasets['test'] = [items | {'mode': 'test', 'label': label_indexer[items['reason']], 'source': 'expert'} for items in all_datasets['expert'] if items['id'] in set(test_document_ids)]

    if 'crowdsourcing' in set(resources):
        all_datasets['train'].extend([items | {'mode': 'train', 'label': label_indexer[items['reason']], 'source': 'crowdsourcing'} for items in all_datasets['crowdsourcing']])

    return all_datasets, label_indexer

def split_to_dev_fold(document_ids, num_fold, index_fold=0):
    full_size = len(document_ids)
    subset_size = full_size // num_fold
    subsets = [document_ids[i * subset_size: (i+1) * subset_size] if i != num_fold -1 else document_ids[(i) * subset_size: ]for i in range(num_fold)]

    dev_index = index_fold
    test_index = 0 if index_fold + 1 == num_fold else index_fold + 1

    dev_set = subsets[dev_index]
    test_set = subsets[test_index]
    train_set = []
    for i, subset in enumerate(subsets):
        if i in set([dev_index, test_index]):
            continue
        else:
            train_set.extend(subset)
    return train_set, dev_set, test_set

def add_special_token(batch_tokens, index_sp):
    for tokens in batch_tokens:
        index = random.randint(0, tokens['input_ids'].shape[1] - 1)
        tokens.data['input_ids'] = torch.cat([tokens.data['input_ids'][0][:index], torch.tensor([index_sp]), tokens.data['input_ids'][0][index:]], dim=0).unsqueeze(0)
        tokens.data['attention_mask'] = torch.cat([tokens.data['attention_mask'][0][:index], torch.tensor([1]), tokens.data['attention_mask'][0][index:]], dim=0).unsqueeze(0)
    return batch_tokens

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', './results/rinna-japanese-gpt2-medium.rand', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', './results/bert-base-japanese-whole-word-masking.rand', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', './results/mbart-large-cc25.rand', 100
    elif mode == 't5-base-encoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.rand', 100
    elif mode == 't5-base-decoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.rand', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', './results/rinna-japanese-roberta-base.rand', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', './results/nlp-waseda-roberta-base-japanese.rand', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', './results/nlp-waseda-roberta-large-japanese.rand', 100
    elif mode == 'nlp-waseda-roberta-base-japanese-with-auto-jumanpp':
        return 'nlp-waseda/roberta-base-japanese-with-auto-jumanpp', './results/nlp-waseda-roberta-base-japanese.rand', 100
    elif mode == 'nlp-waseda-roberta-large-japanese-with-auto-jumanpp':
        return 'nlp-waseda/roberta-large-japanese-with-auto-jumanpp', './results/nlp-waseda-roberta-large-japanese.rand', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', './results/rinna-japanese-gpt-1b.rand', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', './results/xlm-roberta-large.rand', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', './results/xlm-roberta-base.rand', 100

def train_model(run_mode='rinna-gpt2', index_fold=0):
    EPOCHS = 30
    TRAIN_BATCH_SIZE = 16
    INFERENCE_BATCH_SIZE = 32
    WARMUP_STEPS = 0.1
    GRADIENT_ACCUMULATION_STEPS = 16
    DEVICE = 'cuda:0'
    DATASET_MODE = 'crowdsourcing.230205' # crowdsourcing, expert, only_expert, with_crowdsourcing, expert.230205, crowdsourcing.230205
    with_print_logits = False

    model_name, OUTPUT_PATH, _ = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + f'.230413.{seed}'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif 'tohoku' in model_name:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    MAX_SEQUENCE_LENGTH = tokenizer.max_len_single_sentence
    NUM_HEADS = model.config.num_attention_heads

    with_load_weight = False
    if with_load_weight:
        path_weights = './results/xlm-roberta-large.c2c.230226'
        model = AutoModel.from_pretrained(path_weights)
        tokenizer = AutoTokenizer.from_pretrained(path_weights)

    if DATASET_MODE in set(['expert', 'crowdsourcing']):
        datasets, label_indexer = get_data(resource=DATASET_MODE, index_fold=index_fold) # expert, crowdsourcing
    elif DATASET_MODE in set(['only_expert', 'with_crowdsourcing']):
        datasets, label_indexer = get_data_kishimoto(dataset_mode=DATASET_MODE, index_fold=index_fold)
    elif DATASET_MODE in set(['expert.230205', 'crowdsourcing.230205']):
        dataset_mode = {'expert.230205': 'expert', 'crowdsourcing.230205': 'crowd'}
        datasets, label_indexer = get_data_230205(resource=dataset_mode[DATASET_MODE], index_fold=index_fold) # expert.230205, crowdsourcing.230205

    for mode in ['train', 'dev', 'test']:
        datasets[mode] = [items | {'pad_token_id': tokenizer.pad_token_id} for items in datasets[mode]]

    # for k, v in datasets.items():
    #     datasets[k] = tokenize_sentences(v, tokenizer)
    # train_dataset, dev_dataset = split_to_dev(datasets['train'])
    # datasets['train'] = train_dataset
    # datasets['dev'] = dev_dataset

    token_conj = '[CJ]'
    tokenizer.add_special_tokens({'additional_special_tokens': [token_conj]})
    model.resize_token_embeddings(len(tokenizer))
    id_conj = tokenizer._convert_token_to_id_with_added_voc(token_conj)

    s1, y = [tokenizer(d['s1'] + d['s2'], return_tensors='pt') for d in datasets['train']], [d['label'] for d in datasets['train']]
    s1 = add_special_token(s1, id_conj)
    train_dataset = Dataset(s1, y, datasets['train'])
    s1, y = [tokenizer(d['s1'] + d['s2'], return_tensors='pt') for d in datasets['dev']], [d['label'] for d in datasets['dev']]
    s1 = add_special_token(s1, id_conj)
    dev_dataset = Dataset(s1, y, datasets['dev'])
    s1, y = [tokenizer(d['s1'] + d['s2'], return_tensors='pt') for d in datasets['test']], [d['label'] for d in datasets['test']]
    s1 = add_special_token(s1, id_conj)
    test_dataset = Dataset(s1, y, datasets['test'])

    def collate_fn(examples):
        padding_id = examples[0][-1]['pad_token_id'] # model.tokenizer.pad_token_id
        s1 = [torch.as_tensor(example[0].input_ids[0], dtype=torch.long) for example in examples]
        s1 = torch.nn.utils.rnn.pad_sequence(s1, batch_first=True, padding_value=padding_id)
        y = torch.as_tensor([example[1] for example in examples], dtype=torch.long)
        attention_masks1 = [torch.as_tensor(example[0].attention_mask[0], dtype=torch.long) for example in examples]
        attention_masks1 = torch.nn.utils.rnn.pad_sequence(attention_masks1, batch_first=True, padding_value=0)

        if MAX_SEQUENCE_LENGTH != -1:
            if s1.size(1) > MAX_SEQUENCE_LENGTH:
                s1 = s1[:, :MAX_SEQUENCE_LENGTH]
                attention_masks1 = attention_masks1[:, :MAX_SEQUENCE_LENGTH]
        return s1, y, attention_masks1, examples

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    classifier = Classifier(model.config.hidden_size, len(label_indexer))

    num_training_steps = int(EPOCHS*len(train_dataloader)/GRADIENT_ACCUMULATION_STEPS) + 1
    optimizer = torch.optim.AdamW(params=list(model.parameters()) + list(classifier.parameters()), lr=5e-6 if 'xlm' in model_name else 2e-5, weight_decay=0.1 if 'xlm' in model_name else 0.01, betas=(0.9, 0.98) if 'xlm' in model_name else (0.9, 0.99))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS * num_training_steps, num_training_steps=num_training_steps)
    # optimizer = DAdaptAdam(params=list(model.parameters()) + list(classifier.parameters()), lr=1.0, weight_decay=4.0, decouple=True)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS * num_training_steps, num_training_steps=num_training_steps)
    loss_func = torch.nn.CrossEntropyLoss()
    vw = ValueWatcher()

    accelerator = None # Accelerator(mixed_precision=MIXED_PRECISION)
    DEVICE = torch.device('cuda:0') if accelerator is None else accelerator.device
    model = model.to(DEVICE)
    classifier = classifier.to(DEVICE)
    if accelerator is not None:
        model, classifier, optimizer, train_dataloader, dev_dataloader, test_dataloader = accelerator.prepare(model, classifier, optimizer, train_dataloader, dev_dataloader, test_dataloader)

    result_lines = []
    loss = None
    save_files = []
    for e in range(EPOCHS):
        train_total_loss, running_loss = [], []
        model.train()
        classifier.train()
        for i, (s1, y , m1, allx) in enumerate(train_dataloader):
            s1, y, m1 = s1.to(DEVICE), y.to(DEVICE), m1.to(DEVICE)
            outputs = model(s1, attention_mask=m1, output_hidden_states=True, decoder_input_ids=s1) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s1, attention_mask=m1, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
            h1 = torch.stack([h[item.index(id_conj)] for h, item in zip(h1, s1.tolist())])
            logits = classifier(h1)

            loss = loss_func(logits, y)
            if accelerator is None:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
            else:
                accelerator.backward(loss / GRADIENT_ACCUMULATION_STEPS)
            train_total_loss.append(loss.item())

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if optimizer.param_groups[0]['lr'] != 0.0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss = None
                if scheduler is not None:
                    scheduler.step()
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        if loss is not None and optimizer.param_groups[0]['lr'] != 0.0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        # DEV
        with torch.inference_mode():
            model.eval()
            classifier.eval()
            dev_total_loss = []
            dev_tt, dev_ff = 0, 0
            total_dev_y, total_dev_pred = [], []
            dev_predictions = []
            for i, (s1, y , m1, allx) in enumerate(dev_dataloader):
                s1, y, m1 = s1.to(DEVICE), y.to(DEVICE), m1.to(DEVICE)
                outputs = model(s1, attention_mask=m1, output_hidden_states=True, decoder_input_ids=s1) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s1, attention_mask=m1, output_hidden_states=True)
                h1 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
                h1 = torch.stack([h[item.index(id_conj)] for h, item in zip(h1, s1.tolist())])
                logits = classifier(h1)

                loss = loss_func(logits, y)
                dev_total_loss.append(loss.item())

                predicted_label = torch.argmax(logits, dim=1)
                total_dev_y.extend(y.tolist())
                total_dev_pred.extend(predicted_label.tolist())
                dev_tt += sum(predicted_label == y).item()
                dev_ff += y.shape[0] - sum(predicted_label == y).item()

                dev_predictions.extend([items[2] | {'predicted_label': {v: k for k, v in label_indexer.items()}[pred], 'predicted_index': pred, 'logits': logit, 'last_hidden_state': hidden} for items, pred, logit, hidden in zip(allx, predicted_label.tolist(), logits.tolist(), h1.tolist())])

                if with_print_logits:
                    print(f'{logits.item()}, {predicted_label}, {y}')

            dev_rets = get_metrics_scores(total_dev_y, total_dev_pred, label_indexer)
            dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
            dev_acc = dev_tt / (dev_tt + dev_ff)

            # TEST
            test_total_loss = []
            test_tt, test_ff = 0, 0
            total_test_y, total_test_pred = [], []
            test_predictions = []
            for i, (s1, y , m1, allx) in enumerate(test_dataloader):
                s1, y, m1 = s1.to(DEVICE), y.to(DEVICE), m1.to(DEVICE)
                outputs = model(s1, attention_mask=m1, output_hidden_states=True, decoder_input_ids=s1) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s1, attention_mask=m1, output_hidden_states=True)
                h1 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
                h1 = torch.stack([h[item.index(id_conj)] for h, item in zip(h1, s1.tolist())])
                logits = classifier(h1)

                loss = loss_func(logits, y)
                test_total_loss.append(loss.item())

                predicted_label = torch.argmax(logits, dim=1)
                total_test_y.extend(y.tolist())
                total_test_pred.extend(predicted_label.tolist())
                test_tt += sum(predicted_label == y).item()
                test_ff += y.shape[0] - sum(predicted_label == y).item()

                test_predictions.extend([items[2] | {'predicted_label': {v: k for k, v in label_indexer.items()}[pred], 'predicted_index': pred, 'logits': logit, 'last_hidden_state': hidden} for items, pred, logit, hidden in zip(allx, predicted_label.tolist(), logits.tolist(), h1.tolist())])

                if with_print_logits:
                    print(f'{logits.item()}, {predicted_label}, {y}')

            test_rets = get_metrics_scores(total_test_y, total_test_pred, label_indexer)
            test_total_loss = sum(test_total_loss) / len(test_total_loss)
            test_acc = test_tt / (test_tt + test_ff)

        metrics_score = dev_rets['semi_total']['f1']
        if accelerator is None:
            vw.update(metrics_score)
            if vw.is_updated():
                model_file = Path(f'{OUTPUT_PATH}/{run_mode.replace("/", ".")}.fold{index_fold}.{e}.pt')
                with model_file.open('wb') as f:
                    torch.save({'model': model.to('cpu').state_dict(), 'classifier': classifier.to('cpu').state_dict(), 'label_indexer': label_indexer, 'test_metrics': test_rets, 'dev_metrics': dev_rets, 'test_results': test_predictions, 'dev_results': dev_predictions}, f)
                    model, classifier = model.to(DEVICE), classifier.to(DEVICE)

                save_files.append([metrics_score, model_file])

            dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, test_loss: {test_total_loss}, dev_f1: {dev_rets["semi_total"]["f1"]}, test_f1: {test_rets["semi_total"]["f1"]}')
            keys = ['f1', 'precision', 'recall', 'tp', 'fp', 'fn']

            total_metrics = [dev_total_loss, test_total_loss] + [dev_rets['semi_total'][key] for key in keys] + [test_rets['semi_total'][key] for key in keys] + [dev_rets['total'][key] for key in keys] + [test_rets['total'][key] for key in keys]
            dev_metrics = [items[key] for key in keys for label, items in dev_rets.items() if label not in  set(['semi_total', 'total'])]
            test_metrics = [items[key] for key in keys for label, items in test_rets.items() if label not in set(['semi_total', 'total'])]
            result_lines.append([e, train_total_loss] + total_metrics + dev_metrics + test_metrics)

            with Path(f'{OUTPUT_PATH}/result.conj.{run_mode.replace("/", ".")}.fold{index_fold}.csv').open('w') as f:
                f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'test_loss'] + ['dev_semi_total_' + key for key in keys] + ['test_semi_total_' + key for key in keys] + ['dev_total_' + key for key in keys] + ['test_total_' + key for key in keys] + [f'dev_{label}_' + key for label in label_indexer.keys() for key in keys] + [f'test_{label}_' + key for label in label_indexer.keys() for key in keys]))
                f.write('\n')
                for line in result_lines:
                    f.write(','.join(map(str, line)))
                    f.write('\n')
        else:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                vw.update(metrics_score)
                if vw.is_updated():
                    model_file = Path(f'{OUTPUT_PATH}/{run_mode.replace("/", ".")}.fold{index_fold}.{e}.pt')
                    with model_file.open('wb') as f:
                        accelerator.save({'model': accelerator.unwrap_model(model).state_dict(), 'classifier': accelerator.unwrap_model(classifier).state_dict(), 'label_indexer': label_indexer, 'test_metrics': test_rets, 'dev_metrics': dev_rets, 'test_results': test_predictions, 'dev_results': dev_predictions}, f)
                    save_files.append([metrics_score, model_file])

                dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                print(f'{dt_now}, e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, test_loss: {test_total_loss}, dev_f1: {dev_rets["total"]["f1"]}, test_f1: {test_rets["total"]["f1"]}')
                keys = ['f1', 'precision', 'recall', 'tp', 'fp', 'fn']

                total_metrics = [dev_total_loss, test_total_loss] + [dev_rets['semi_total'][key] for key in keys] + [test_rets['semi_total'][key] for key in keys] + [dev_rets['total'][key] for key in keys] + [test_rets['total'][key] for key in keys]
                dev_metrics = [items[key] for key in keys for label, items in dev_rets.items() if label not in set(['semi_total', 'total'])]
                test_metrics = [items[key] for key in keys for label, items in test_rets.items() if label not in set(['semi_total', 'total'])]
                result_lines.append([e, train_total_loss] + total_metrics + dev_metrics + test_metrics)

                with Path(f'{OUTPUT_PATH}/result.conj.{run_mode.replace("/", ".")}.fold{index_fold}.csv').open('w') as f:
                    f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'test_loss'] + ['dev_semi_total_' + key for key in keys] + ['test_semi_total_' + key for key in keys] + ['dev_total_' + key for key in keys] + ['test_total_' + key for key in keys] + [f'dev_{label}_' + key for label in label_indexer.keys() for key in keys] + [f'test_{label}_' + key for label in label_indexer.keys() for key in keys]))
                    f.write('\n')
                    for line in result_lines:
                        f.write(','.join(map(str, line)))
                        f.write('\n')
        
    for item in save_files:
        if item[0] != vw.max_score:
            item[1].unlink(missing_ok=False)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'

    # get_data(resource='crowdsourcing')
    is_single = False
    # run_modes = ['xlm-roberta-base']
    # run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'nlp-waseda-roberta-base-japanese-with-auto-jumanpp', 'nlp-waseda-roberta-large-japanese-with-auto-jumanpp', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    # run_modes = ['xlm-roberta-large']
    if is_single:
        train_model(run_modes[-2], index_fold=0)
    else:
        for run_mode in run_modes:
            for i in range(5):
                train_model(run_mode, index_fold=i)



    

