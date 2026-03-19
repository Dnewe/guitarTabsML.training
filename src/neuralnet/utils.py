import numpy as np


def get_accuracy_note(targets, logits):
    mask = targets != 0
    correct = np.sum((targets == (logits>0.5)) & mask)
    return correct / np.sum(mask)


def get_accuracy_beat(targets, logits):
    logits = logits >0.5
    mask = targets != 0  
    logits_masked = [row[mask_row] for row, mask_row in zip(logits, mask)]
    targets_masked = [row[mask_row] for row, mask_row in zip(targets, mask)]
    row_correct = [np.array_equal(t, y) for t, y in zip(targets_masked, logits_masked)]
    return float(np.mean(row_correct))


def loss_CCE(targets, logits):
    loss = -np.sum(targets * np.log(logits+1e-8)) / logits.shape[0]
    return loss
    
    
def loss_BCE(targets, logits):
    logits = np.clip(logits, 1e-8, 1-1e-8)
    res = -np.mean(targets * np.log(logits)
                + (1-targets) * np.log(1-logits))
    return res