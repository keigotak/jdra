import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score

def get_confusion_matrix1(y_true, y_pred, num_class):
    N = num_class
    y_true = torch.as_tensor(y_true, dtype=torch.long, device='cpu')
    y_pred = torch.as_tensor(y_pred, dtype=torch.long, device='cpu')
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]),
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])
    ).to_dense().tolist()

def get_confusion_matrix2(y_true, y_pred, num_class):
    N = num_class
    y_true = torch.as_tensor(y_true, dtype=torch.long, device='cpu')
    y_pred = torch.as_tensor(y_pred, dtype=torch.long, device='cpu')
    y = N * y_true + y_pred
    check_flgs = [i in y for i in range(N**2)]
    y = torch.bincount(y)
    for i, check_flg in enumerate(check_flgs):
        if i < y.shape[0]:
            if not check_flg and y[i] != 0:
                y = torch.cat([y[:i], torch.zeros(1, dtype=torch.long, device='cpu'), y[i:]])
        else:
            if not check_flg:
                y = torch.cat([y, torch.zeros(1, dtype=torch.long, device='cpu')])
    y = y.reshape(N, N)
    return y.tolist()

def get_metrics_scores(y_true, y_pred, label_indexer):
    cm = get_confusion_matrix2(y_true, y_pred, len(label_indexer))
    rets = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    semi_total_tp, semi_total_fp, semi_total_fn = 0, 0, 0
    for k, i in label_indexer.items():
        tp = cm[i][i]
        fp = sum([element for j, element in enumerate(cm[i]) if j != i])
        fn = sum([row[i] for j, row in enumerate(cm) if j != i])

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0.0 else 0.0
        rets[k] = {'index': i, 'tp': tp, 'fp': fp, 'fn': fn, 'total': tp+fn, 'precision': precision, 'recall': recall, 'f1': f1}

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if k != '談話関係なし':
            semi_total_tp += tp
            semi_total_fp += fp
            semi_total_fn += fn

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0.0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) != 0.0 else 0.0
    rets['total'] = {'index': None, 'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'total': total_tp+total_fn, 'precision': total_precision, 'recall': total_recall, 'f1': total_f1}

    semi_total_precision = semi_total_tp / (semi_total_tp + semi_total_fp) if (semi_total_tp + semi_total_fp) != 0 else 0.0
    semi_total_recall = semi_total_tp / (semi_total_tp + semi_total_fn) if (semi_total_tp + semi_total_fn) != 0 else 0.0
    semi_total_f1 = 2 * semi_total_precision * semi_total_recall / (semi_total_precision + semi_total_recall) if (semi_total_precision + semi_total_recall) != 0.0 else 0.0
    rets['semi_total'] = {'index': None, 'tp': semi_total_tp, 'fp': semi_total_fp, 'fn': semi_total_fn, 'total': semi_total_tp+semi_total_fn, 'precision': semi_total_precision, 'recall': semi_total_recall, 'f1': semi_total_f1}

    return rets

def get_metrics_scores_by_divided_type(y_true, y_pred, label_indexer):
    cm = get_confusion_matrix2(y_true, y_pred, len(label_indexer))
    rets = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    semi_total_tp, semi_total_fp, semi_total_fn = 0, 0, 0
    for k, i in label_indexer.items():
        tp = cm[i][i]
        fp = sum([element for j, element in enumerate(cm[i]) if j != i])
        fn = sum([row[i] for j, row in enumerate(cm) if j != i])

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0.0 else 0.0
        rets[k] = {'index': i, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if k != '談話関係なし':
            semi_total_tp += tp
            semi_total_fp += fp
            semi_total_fn += fn

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0.0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) != 0.0 else 0.0
    rets['total'] = {'index': None, 'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'precision': total_precision, 'recall': total_recall, 'f1': total_f1}

    semi_total_precision = semi_total_tp / (semi_total_tp + semi_total_fp) if (semi_total_tp + semi_total_fp) != 0 else 0.0
    semi_total_recall = semi_total_tp / (semi_total_tp + semi_total_fn) if (semi_total_tp + semi_total_fn) != 0 else 0.0
    semi_total_f1 = 2 * semi_total_precision * semi_total_recall / (semi_total_precision + semi_total_recall) if (semi_total_precision + semi_total_recall) != 0.0 else 0.0
    rets['semi_total'] = {'index': None, 'tp': semi_total_tp, 'fp': semi_total_fp, 'fn': semi_total_fn, 'precision': semi_total_precision, 'recall': semi_total_recall, 'f1': semi_total_f1}

    return rets

