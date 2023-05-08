import re
import pandas as pd
import datetime

import os
import random

import numpy as np
import torch
import torch.nn as nn

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path
import transformers
transformers.BertTokenizer = transformers.BertJapaneseTokenizer
transformers.trainer_utils.set_seed(SEED)

from accelerate import Accelerator
from transformers import T5Tokenizer, T5Model
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup



from BertJapaneseTokenizerFast import BertJapaneseTokenizerFast
from ValueWatcher import ValueWatcher
from Helperfunctions import get_metrics_scores

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, num_class, pooling_method='mean'):
        super(Classifier, self).__init__()
        self.encoder_f = nn.ModuleList([nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)])
        self.encoder_b = nn.ModuleList([nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)])
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.linear = torch.nn.Linear(2 * hidden_size, num_class)
        self.num_layers = num_layers
        self.pooling_method = pooling_method
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x, mask):
        for i in range(self.num_layers):
            for enc in [self.encoder_f[i], self.encoder_b[i]]:
                encoder_outputs, hidden = enc(x)
                x = encoder_outputs + x
                x = x[torch.arange(x.shape[0]-1, -1, -1), :, :]
                x = self.dropout(x)

        x, weights = self.attention(x, x, x, key_padding_mask=mask==0)
        x = self.dropout(x)

        if self.pooling_method == 'mean':
            x = self.pooling_mean(x, mask)
        elif self.pooling_method == 'max':
            x = self.pooling_max(x, mask)
        return x
    
    def pooling_mean(self, x, mask):
        mask = torch.argmin(mask, dim=1, keepdim=True)
        mask, idx = torch.sort(mask, dim=0)
        x = torch.gather(x, dim=0, index=idx.unsqueeze(2).expand(-1, x.shape[1], x.shape[2]))
        unique_items, counts = torch.unique_consecutive(mask, return_counts=True)
        unique_items = unique_items.tolist()
        counts = [0] + torch.cumsum(counts, -1).tolist()
        x = torch.cat([torch.mean(x[counts[i]: counts[i+1], :ui, :], dim=1) if ui != 0 else torch.mean(x[counts[i]: counts[i+1], :, :], dim=1) for i, ui in enumerate(unique_items)])
        idx = torch.argsort(idx, dim=0)
        x = torch.gather(x, dim=0, index=idx.expand(-1, x.shape[1]))
        return x

    def pooling_max(self, x, mask):
        mask = torch.argmin(mask, dim=1, keepdim=True)
        mask, idx = torch.sort(mask, dim=0)
        x = torch.gather(x, dim=0, index=idx.unsqueeze(2).expand(-1, x.shape[1], x.shape[2]))
        unique_items, counts = torch.unique_consecutive(mask, return_counts=True)
        unique_items = unique_items.tolist()
        counts = [0] + torch.cumsum(counts, -1).tolist()
        x = torch.cat([torch.max(x[counts[i]: counts[i+1], :ui, :], dim=1)[0] if ui != 0 else torch.max(x[counts[i]: counts[i+1], :, :], dim=1)[0] for i, ui in enumerate(unique_items)])
        idx = torch.argsort(idx, dim=0)
        x = torch.gather(x, dim=0, index=idx.expand(-1, x.shape[1]))
        return x


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

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', './results/rinna-japanese-gpt2-medium.concat', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', './results/bert-base-japanese-whole-word-masking.concat', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', './results/mbart-large-cc25.concat', 100
    elif mode == 't5-base-encoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.concat', 100
    elif mode == 't5-base-decoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.concat', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', './results/rinna-japanese-roberta-base.concat', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', './results/nlp-waseda-roberta-base-japanese.concat', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', './results/nlp-waseda-roberta-large-japanese.concat', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', './results/rinna-japanese-gpt-1b.concat', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', './results/xlm-roberta-large.concat', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', './results/xlm-roberta-base.concat', 100


def tokenize_sentences(dataset, tokenizer):
    _dataset = []
    for items in dataset:
        tokens1 = tokenizer(items['s1'], return_tensors='pt')
        tokens2 = tokenizer(items['s2'], return_tensors='pt')
        _dataset.append(items | {'token1': tokens1, 'token2': tokens2, 'pad_token_id': tokenizer.pad_token_id})
    return _dataset


def get_datasets(path):
    with Path(path).open('r') as f:
        texts = f.readlines()
    texts = [text.strip().split('\t') for text in texts]
    s1, s2, l = [], [], []
    for text in texts:
        t1 = random.choice([text[1], text[2]])
        if t1 == text[1]:
            s1.append(text[1])
            s2.append(text[2])
            l.append(0)
        else:
            s1.append(text[2])
            s2.append(text[1])
            l.append(1)
    return s1, s2, l


def allocate_data_to_device(data, device='cpu'):
    if device != 'cpu':
        return data.to('cuda:0')
    else:
        return data


def train_model(run_mode='rinna-gpt2'):
    EPOCHS = 100
    MIXED_PRECISION = 'fp16'
    TRAIN_BATCH_SIZE = 32
    INFERENCE_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 16
    WARMUP_STEPS = 10
    DEVICE = 'cuda:0'
    NUM_LAYERS = 3
    POOLING_METHOD = 'max'
    DATASET_MODE = 'expert' # crowdsourcing, expert, kishimoto
    with_activation_function = False
    with_print_logits = False

    model_name, OUTPUT_PATH, _ = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.230119.0'
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

    if DATASET_MODE in set(['expert', 'crowdsourcing']):
        datasets, label_indexer = get_data(resource=DATASET_MODE) # expert, crowdsourcing
    elif DATASET_MODE in set(['kishimoto']):
        datasets, label_indexer = get_data_kishimoto()
    for k, v in datasets.items():
        datasets[k] = tokenize_sentences(v, tokenizer)
    train_dataset, dev_dataset = split_to_dev(datasets['train'])
    datasets['train'] = train_dataset
    datasets['dev'] = dev_dataset

    s1, s2, y = [tokenizer(d['s1'], return_tensors='pt') for d in datasets['train']], [tokenizer(d['s2'], return_tensors='pt') for d in datasets['train']], [d['label'] for d in datasets['train']]
    train_dataset = Dataset(s1, s2, y, datasets['train'])
    s1, s2, y = [tokenizer(d['s1'], return_tensors='pt') for d in datasets['dev']], [tokenizer(d['s2'], return_tensors='pt') for d in datasets['dev']], [d['label'] for d in datasets['dev']]
    dev_dataset = Dataset(s1, s2, y, datasets['dev'])
    s1, s2, y = [tokenizer(d['s1'], return_tensors='pt') for d in datasets['test']], [tokenizer(d['s2'], return_tensors='pt') for d in datasets['test']], [d['label'] for d in datasets['test']]
    test_dataset = Dataset(s1, s2, y, datasets['test'])

    def collate_fn(examples):
        padding_id = examples[0][-1]['pad_token_id'] # model.tokenizer.pad_token_id
        s1 = [torch.as_tensor(example[0].input_ids[0], dtype=torch.long) for example in examples]
        s1 = torch.nn.utils.rnn.pad_sequence(s1, batch_first=True, padding_value=padding_id)
        s2 = [torch.as_tensor(example[1].input_ids[0], dtype=torch.long) for example in examples]
        s2 = torch.nn.utils.rnn.pad_sequence(s2, batch_first=True, padding_value=padding_id)
        y = torch.as_tensor([example[2] for example in examples], dtype=torch.long)
        attention_masks1 = [torch.as_tensor(example[0].attention_mask[0], dtype=torch.long) for example in examples]
        attention_masks1 = torch.nn.utils.rnn.pad_sequence(attention_masks1, batch_first=True, padding_value=0)
        attention_masks2 = [torch.as_tensor(example[1].attention_mask[0], dtype=torch.long) for example in examples]
        attention_masks2 = torch.nn.utils.rnn.pad_sequence(attention_masks2, batch_first=True, padding_value=0)

        if MAX_SEQUENCE_LENGTH != -1:
            if s1.size(1) > MAX_SEQUENCE_LENGTH:
                s1 = s1[:, :MAX_SEQUENCE_LENGTH]
                attention_masks1 = attention_masks1[:, :MAX_SEQUENCE_LENGTH]
            if s2.size(1) > MAX_SEQUENCE_LENGTH:
                s2 = s2[:, :MAX_SEQUENCE_LENGTH]
                attention_masks2 = attention_masks2[:, :MAX_SEQUENCE_LENGTH]
        return s1, s2, y, attention_masks1, attention_masks2, examples

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
    classifier = Classifier(model.config.hidden_size, NUM_LAYERS, NUM_HEADS, len(label_indexer), POOLING_METHOD)

    optimizer = torch.optim.AdamW(params=list(model.parameters()) + list(classifier.parameters()), lr=2e-6 if 'xlm' in model_name else 3e-5, weight_decay=0.01)
    scheduler = None # get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=int(EPOCHS*len(train_dataloader)/GRADIENT_ACCUMULATION_STEPS) + 1)
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
        for i, (s1, s2, y , m1, m2, allx) in enumerate(train_dataloader):
            s1, s2, y, m1, m2 = s1.to(DEVICE), s2.to(DEVICE), y.to(DEVICE), m1.to(DEVICE), m2.to(DEVICE)
            outputs = model(s1, attention_mask=m1, output_hidden_states=True, decoder_input_ids=s1) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s1, attention_mask=m1, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
            h1 = classifier(h1, m1)

            outputs = model(s2, attention_mask=m2, output_hidden_states=True, decoder_input_ids=s2) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s2, attention_mask=m2, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
            h2 = classifier(h2, m2)

            logits = classifier.linear(torch.cat([h1, h2], dim=1))
            loss = loss_func(logits, y)
            if accelerator is None:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
            else:
                accelerator.backward(loss / GRADIENT_ACCUMULATION_STEPS)
            train_total_loss.append(loss.item())

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss = None
                if scheduler is not None:
                    scheduler.step()
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        if loss is not None:
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
            for i, (s1, s2, y , m1, m2, allx) in enumerate(dev_dataloader):
                s1, s2, y, m1, m2 = s1.to(DEVICE), s2.to(DEVICE), y.to(DEVICE), m1.to(DEVICE), m2.to(DEVICE)
                outputs = model(s1, attention_mask=m1, output_hidden_states=True, decoder_input_ids=s1) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s1, attention_mask=m1, output_hidden_states=True)
                h1 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
                h1 = classifier(h1, m1)

                outputs = model(s2, attention_mask=m2, output_hidden_states=True, decoder_input_ids=s2) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s2, attention_mask=m2, output_hidden_states=True)
                h2 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
                h2 = classifier(h2, m2)

                logits = classifier.linear(torch.cat([h1, h2], dim=1))
                loss = loss_func(logits, y)
                dev_total_loss.append(loss.item())

                predicted_label = torch.argmax(logits, dim=1)
                total_dev_y.extend(y.tolist())
                total_dev_pred.extend(predicted_label.tolist())
                dev_tt += sum(predicted_label == y).item()
                dev_ff += y.shape[0] - sum(predicted_label == y).item()

                if with_print_logits:
                    print(f'{logits.item()}, {predicted_label}, {y}')

            dev_rets = get_metrics_scores(total_dev_y, total_dev_pred, label_indexer)
            dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
            dev_acc = dev_tt / (dev_tt + dev_ff)

            # TEST
            test_total_loss = []
            test_tt, test_ff = 0, 0
            total_test_y, total_test_pred = [], []
            for i, (s1, s2, y , m1, m2, allx) in enumerate(test_dataloader):
                s1, s2, y, m1, m2 = s1.to(DEVICE), s2.to(DEVICE), y.to(DEVICE), m1.to(DEVICE), m2.to(DEVICE)
                outputs = model(s1, attention_mask=m1, output_hidden_states=True, decoder_input_ids=s1) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s1, attention_mask=m1, output_hidden_states=True)
                h1 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
                h1 = classifier(h1, m1)

                outputs = model(s2, attention_mask=m2, output_hidden_states=True, decoder_input_ids=s2) if run_mode in set(['t5-base-encoder', 't5-base-decoder']) else model(s2, attention_mask=m2, output_hidden_states=True)
                h2 = outputs.encoder_last_hidden_state if run_mode in set(['t5-base-encoder']) else outputs.last_hidden_state
                h2 = classifier(h2, m2)

                logits = classifier.linear(torch.cat([h1, h2], dim=1))
                loss = loss_func(logits, y)
                test_total_loss.append(loss.item())

                predicted_label = torch.argmax(logits, dim=1)
                total_test_y.extend(y.tolist())
                total_test_pred.extend(predicted_label.tolist())
                test_tt += sum(predicted_label == y).item()
                test_ff += y.shape[0] - sum(predicted_label == y).item()

                if with_print_logits:
                    print(f'{logits.item()}, {predicted_label}, {y}')

            test_rets = get_metrics_scores(y, predicted_label, label_indexer)
            test_total_loss = sum(test_total_loss) / len(test_total_loss)
            test_acc = test_tt / (test_tt + test_ff)

        metrics_score = dev_rets['semi_total']['f1']
        if accelerator is None:
            vw.update(metrics_score)
            if vw.is_updated():
                model_file = Path(f'{OUTPUT_PATH}/{model_name.replace("/", ".")}.{e}.pt')
                with model_file.open('wb') as f:
                    if accelerator is None:
                        torch.save({'model': model.to('cpu').state_dict(), 'classifier': classifier.to('cpu').state_dict()}, f)
                        model, classifier = model.to(DEVICE), classifier.to(DEVICE)
                    else:
                        accelerator.save({'model': accelerator.unwrap_model(model).state_dict(), 'classifier': accelerator.unwrap_model(classifier).state_dict()}, f)

                save_files.append([metrics_score, model_file])

            dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, test_loss: {test_total_loss}, dev_f1: {dev_rets["semi_total"]["f1"]}, test_f1: {test_rets["semi_total"]["f1"]}')
            keys = ['f1', 'precision', 'recall', 'tp', 'fp', 'fn']

            total_metrics = [dev_total_loss, test_total_loss] + [dev_rets['semi_total'][key] for key in keys] + [test_rets['semi_total'][key] for key in keys] + [dev_rets['total'][key] for key in keys] + [test_rets['total'][key] for key in keys]
            dev_metrics = [items[key] for key in keys for label, items in dev_rets.items() if label not in  set(['semi_total', 'total'])]
            test_metrics = [items[key] for key in keys for label, items in test_rets.items() if label not in set(['semi_total', 'total'])]
            result_lines.append([e, train_total_loss] + total_metrics + dev_metrics + test_metrics)

            with Path(f'{OUTPUT_PATH}/result.concat.{model_name.replace("/", ".")}.csv').open('w') as f:
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
                    model_file = Path(f'{OUTPUT_PATH}/{model_name.replace("/", ".")}.{e}.pt')
                    with model_file.open('wb') as f:
                        accelerator.save({'model': accelerator.unwrap_model(model).state_dict(), 'classifier': accelerator.unwrap_model(classifier).state_dict()}, f)
                    save_files.append([metrics_score, model_file])

                dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, test_loss: {test_total_loss}, dev_f1: {dev_rets["total"]["f1"]}, test_f1: {test_rets["total"]["f1"]}')
                keys = ['f1', 'precision', 'recall', 'tp', 'fp', 'fn']

                total_metrics = [dev_total_loss, test_total_loss] + [dev_rets['semi_total'][key] for key in keys] + [test_rets['semi_total'][key] for key in keys] + [dev_rets['total'][key] for key in keys] + [test_rets['total'][key] for key in keys]
                dev_metrics = [items[key] for key in keys for label, items in dev_rets.items() if label not in set(['semi_total', 'total'])]
                test_metrics = [items[key] for key in keys for label, items in test_rets.items() if label not in set(['semi_total', 'total'])]
                result_lines.append([e, train_total_loss] + total_metrics + dev_metrics + test_metrics)

                with Path(f'{OUTPUT_PATH}/result.concat.{model_name.replace("/", ".")}.csv').open('w') as f:
                    f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'test_loss'] + ['dev_semi_total_' + key for key in keys] + ['test_semi_total_' + key for key in keys] + ['dev_total_' + key for key in keys] + ['test_total_' + key for key in keys] + [f'dev_{label}_' + key for label in label_indexer.keys() for key in keys] + [f'test_{label}_' + key for label in label_indexer.keys() for key in keys]))
                    f.write('\n')
                    for line in result_lines:
                        f.write(','.join(map(str, line)))
                        f.write('\n')
        
    for item in save_files:
        if item[0] != vw.max_score:
            item[1].unlink(missing_ok=False)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    is_single = False
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    if is_single:
        train_model(run_modes[1])
    else:
        for run_mode in run_modes:
            train_model(run_mode)



    

