import re
import pandas as pd

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

from transformers import T5Tokenizer, T5Model
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
# from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import BertModel
from BertJapaneseTokenizerFast import BertJapaneseTokenizerFast
from ValueWatcher import ValueWatcher

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
                numbers, reason = items[0].split('-'), items[1]
                sentence_index_from = int(numbers[0])
                sentence_index_to = int(numbers[1])

                if resource == 'crowdsourcing':
                    reason = reason.split(' ')[0].split(':')[0]
                discrosures.append([sentences[sentence_index_from - 1][0], sentences[sentence_index_from - 1][2], sentences[sentence_index_to - 1][2], sentence_index_from, sentence_index_to, reason])
            else:
                items = text.split(' ')
                sentences.append([problem_index, items[0], items[1]])    

    all_datasets = {}
    label_indexer = {}
    for mode in modes:
        datasets = []
        _ids = set(ids[mode])
        for p in problems:
            if p[0] in _ids:
                datasets.append(p + [mode])
            else:
                pass
            if p[5] not in label_indexer.keys():
                label_indexer[p[5]] = len(label_indexer)
        all_datasets[mode] = datasets
    
    for mode in modes:
        for i in range(len(all_datasets[mode])):
            all_datasets[mode][i].append(label_indexer[all_datasets[mode][i][5]])

    return all_datasets, label_indexer

def split_to_dev(data):
    train_size = int(0.8 * len(data))
    return data[:train_size], data[train_size:]

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', './results/rinna-japanese-gpt2-medium.doublet', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', './results/bert-base-japanese-whole-word-masking.doublet', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', './results/mbart-large-cc25.doublet', 100
    elif mode == 't5-base':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.doublet', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', './results/rinna-japanese-roberta-base.doublet', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', './results/nlp-waseda-roberta-base-japanese.doublet', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', './results/nlp-waseda-roberta-large-japanese.doublet', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', './results/rinna-japanese-gpt-1b.doublet', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', './results/xlm-roberta-large.doublet', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', './results/xlm-roberta-base.doublet', 100


def tokenize_sentences(dataset, tokenizer):
    _dataset = []
    for items in dataset:
        tokens1 = tokenizer(items[1], return_tensors='pt')
        tokens2 = tokenizer(items[2], return_tensors='pt')
        _dataset.append(items + [tokens1, tokens2])
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
    BATCH_SIZE = 32
    WARMUP_STEPS = int(1000 // BATCH_SIZE * 0.1)
    DEVICE = 'cuda:0'
    with_activation_function = False
    with_print_logits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
    OUTPUT_PATH = OUTPUT_PATH + '.230103.0'
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    print(run_mode)
    print(OUTPUT_PATH)

    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    # elif 'mbart' in model_name:
    #     model = MBartForConditionalGeneration.from_pretrained(model_name)
    #     tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
    elif 't5' in model_name:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif 'tohoku' in model_name:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = allocate_data_to_device(model, DEVICE)

    datasets, label_indexer = get_data(resource='expert') # expert, crowdsourcing
    for k, v in datasets.items():
        datasets[k] = tokenize_sentences(v, tokenizer)
    train_dataset, dev_dataset = split_to_dev(datasets['train'])
    datasets['train'] = train_dataset
    datasets['dev'] = dev_dataset

    output_layer = allocate_data_to_device(torch.nn.Linear(2 * model.config.hidden_size, len(label_indexer)), DEVICE)
    activation_function = torch.nn.SELU()
    optimizer = AdamW(params=list(model.parameters()) + list(output_layer.parameters()), lr=2e-6 if 'xlm' in model_name else 2e-5, weight_decay=0.01)
    # optimizer = AdamW(params=list(output_layer.parameters()), lr=2e-5, weight_decay=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCEWithLogitsLoss()
    vw = ValueWatcher()

    result_lines = []
    for e in range(100):
        train_total_loss, running_loss = [], []
        model.train()
        output_layer.train()
        for d in datasets['train']:
            inputs = d[8]
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            # o1 = output_layer(h1)
            # o1 = activation_function(o1) if with_activation_function else o1

            inputs = d[9]
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)

            logits = output_layer(torch.cat([h1, h2], dim=1))
            logits = activation_function(logits) if with_activation_function else logits

            loss = loss_func(logits, torch.as_tensor([d[7]], dtype=torch.long, device=DEVICE))

            train_total_loss.append(loss.item())
            running_loss.append(loss)
            if len(running_loss) >= BATCH_SIZE:
                running_loss = torch.mean(torch.stack(running_loss), dim=0)
                optimizer.zero_grad(set_to_none=True)
                running_loss.backward()
                optimizer.step()
                running_loss = []
        train_total_loss = sum(train_total_loss) / len(train_total_loss)
        if len(running_loss) > 0:
            running_loss = torch.mean(torch.stack(running_loss), dim=0)
            optimizer.zero_grad(set_to_none=True)
            running_loss.backward()
            optimizer.step()
            running_loss = []

        # DEV
        model.eval()
        output_layer.eval()
        dev_total_loss = []
        dev_tt, dev_ff = 0, 0
        for d in datasets['dev']:
            inputs = d[8]
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            # o1 = output_layer(h1)
            # o1 = activation_function(o1) if with_activation_function else o1

            inputs = d[9]
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            # h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            # o2 = output_layer(h2)
            # o2 = activation_function(o2) if with_activation_function else o2

            logits = output_layer(torch.cat([h1, h2], dim=1))
            logits = activation_function(logits) if with_activation_function else logits
            predicted_label = torch.argmax(logits).item()

            label = d[7]
            loss = loss_func(logits, torch.as_tensor([label], dtype=torch.long, device=DEVICE))
            dev_total_loss.append(loss.item())

            if predicted_label == d[7]:
                dev_tt += 1
            else:
                dev_ff += 1

            if with_print_logits:
                print(f'{logits.item()}, {predicted_label}, {label}')

        dev_total_loss = sum(dev_total_loss) / len(dev_total_loss)
        dev_acc = dev_tt / (dev_tt + dev_ff)

        vw.update(dev_acc)
        if vw.is_updated():
            with Path(f'{OUTPUT_PATH}/{model_name.replace("/", ".")}.{e}.pt').open('wb') as f:
                torch.save({'model': model.state_dict(), 'output_layer': output_layer.state_dict()}, f)

        # TEST
        test_total_loss = []
        test_tt, test_ff = 0, 0
        for d in datasets['test']:
            inputs = d[8]
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            # h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            h1 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h1 = torch.mean(h1, dim=1)
            # o1 = output_layer(h1)
            # o1 = activation_function(o1) if with_activation_function else o1

            inputs = d[9]
            inputs = {k: allocate_data_to_device(inputs[k], DEVICE) for k in inputs.keys() if k != 'offset_mapping'}
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=inputs['input_ids']) if 'mbart' in model_name or 't5' in model_name else model(**inputs, output_hidden_states=True)
            # h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name else outputs.last_hidden_state
            h2 = outputs.encoder_last_hidden_state if 'mbart' in model_name or 't5' in model_name else outputs.last_hidden_state
            h2 = torch.mean(h2, dim=1)
            # o2 = output_layer(h2)
            # o2 = activation_function(o2) if with_activation_function else o2

            logits = output_layer(torch.cat([h1, h2], dim=1))
            logits = activation_function(logits) if with_activation_function else logits
            predicted_label = torch.argmax(logits).item()

            label = d[7]
            loss = loss_func(logits, torch.as_tensor([label], dtype=torch.long, device=DEVICE))
            test_total_loss.append(loss.item())

            if predicted_label == d[7]:
                test_tt += 1
            else:
                test_ff += 1

            if with_print_logits:
                print(f'{logits.item()}, {predicted_label}, {label}')

        test_total_loss = sum(test_total_loss) / len(test_total_loss)
        test_acc = test_tt / (test_tt + test_ff)

        print(f'e: {e}, train_loss: {train_total_loss}, dev_loss: {dev_total_loss}, dev_acc: {dev_acc}, test_loss: {test_total_loss}, test_acc: {test_acc}')
        result_lines.append([e, train_total_loss, dev_total_loss, dev_acc, test_total_loss, test_acc])

    with Path(f'{OUTPUT_PATH}/result.doublet.{model_name.replace("/", ".")}.csv').open('w') as f:
        f.write(','.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc', 'test_loss', 'test_acc']))
        f.write('\n')
        for line in result_lines:
            f.write(','.join(map(str, line)))
            f.write('\n')


if __name__ == '__main__':
    # get_data(resource='crowdsourcing')
    is_single = True
    run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
    if is_single:
        train_model(run_modes[1])
    else:
        for run_mode in run_modes[1:]:
            train_model(run_mode)



    

