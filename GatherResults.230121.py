import pandas as pd
from pathlib import Path

def get_properties(mode):
    if mode == 'rinna-gpt2':
        return 'rinna/japanese-gpt2-medium', './results/rinna-japanese-gpt2-medium.conj', 100
    elif mode == 'tohoku-bert':
        return 'cl-tohoku/bert-base-japanese-whole-word-masking', './results/bert-base-japanese-whole-word-masking.conj', 100
    elif mode == 'mbart':
        return 'facebook/mbart-large-cc25', './results/mbart-large-cc25.conj', 100
    elif mode == 't5-base-encoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.conj', 100
    elif mode == 't5-base-decoder':
        return 'megagonlabs/t5-base-japanese-web', './results/t5-base-japanese-web.conj', 100
    elif mode =='rinna-roberta':
        return 'rinna/japanese-roberta-base', './results/rinna-japanese-roberta-base.conj', 100
    elif mode == 'nlp-waseda-roberta-base-japanese':
        return 'nlp-waseda/roberta-base-japanese', './results/nlp-waseda-roberta-base-japanese.conj', 100
    elif mode == 'nlp-waseda-roberta-large-japanese':
        return 'nlp-waseda/roberta-large-japanese', './results/nlp-waseda-roberta-large-japanese.conj', 100
    elif mode == 'rinna-japanese-gpt-1b':
        return 'rinna/japanese-gpt-1b', './results/rinna-japanese-gpt-1b.conj', 100
    elif mode == 'xlm-roberta-large':
        return 'xlm-roberta-large', './results/xlm-roberta-large.conj', 100
    elif mode == 'xlm-roberta-base':
        return 'xlm-roberta-base', './results/xlm-roberta-base.conj', 100

MODE_LIST = ['rinna-gpt2', 'tohoku-bert', 't5-base-encoder', 't5-base-decoder', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'nlp-waseda-roberta-large-japanese', 'rinna-japanese-gpt-1b', 'xlm-roberta-large', 'xlm-roberta-base']
TAG = '.230120.0'
METHOD = '.conj'

def gather():
    rets = []
    for mode in MODE_LIST:
        model_name, OUTPUT_PATH, _ = get_properties(mode)
        OUTPUT_PATH = OUTPUT_PATH + TAG + f'/result{METHOD}.{model_name.replace("/", ".")}.csv'
        df = pd.read_csv(OUTPUT_PATH, header=0, index_col=None).sort_values('dev_semi_total_f1', ascending=False)
        df['model_name'] = [model_name for _ in range(df.shape[0])]
        df['output_path'] = [OUTPUT_PATH for _ in range(df.shape[0])]
        if len(rets) == 0:
            rets.append(df.columns.tolist())
        rets.extend(df[df['dev_semi_total_f1'] == df['dev_semi_total_f1'].max()].values.tolist())

    df = pd.DataFrame(rets[1:], columns=rets[0])
    df.to_csv(f'results{TAG}{METHOD}.csv', index=None)
    print(rets)

if __name__=='__main__':
    gather()
