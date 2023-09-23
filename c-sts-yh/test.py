import json
from pathlib import Path

from datasets import load_dataset, disable_caching
from torch.utils.data import DataLoader

disable_caching()

output_dir = "/home/somebodil/workspace/private-projects/Sentence-Representation/c-sts/output/princeton-nlp__sup-simcse-roberta-large/hypernet__enc_bi_encoder__lr_1e-5__wd_0.1__trans_False__obj_mse__tri_hadamard__s_42__finetune_True__val_as_test_True"
prefix = "test"
label_diff_threshold = 0.75
path = Path(output_dir, f"{prefix}_examples.csv")
dataset = load_dataset("csv", data_files={"train": str(path.absolute())}, split="train")

dataloader = DataLoader(dataset, batch_size=2)

examples_correct = []
idx_correct = 1
examples_incorrect = []
idx_incorrect = 1
for batch in dataloader:
    s1, s2, c_high, c_low, label_high, label_low, pred_high, pred_low = batch['sentence1'][0], batch['sentence2'][0], batch['condition'][0], batch['condition'][1], batch['label'][0], batch['label'][1], batch['pred'][0], batch['pred'][1]
    if batch['sentence1'][0] != batch['sentence1'][1] and batch['sentence2'][0] != batch['sentence2'][1]:
        raise ValueError("Sentence pairs should be same")

    if label_low > label_high:
        label_high, label_low = label_low, label_high
        c_high, c_low = c_low, c_high
        pred_high, pred_low = pred_low, pred_high

    label_diff = label_high - label_low
    if label_diff >= label_diff_threshold:
        if pred_high > pred_low:
            examples_correct.append({
                'idx': idx_correct,
                'sentence1': s1,
                'sentence2': s2,
                'c_high': c_high,
                'c_low': c_low,
                'label_high': label_high.item(),
                'label_low': label_low.item(),
                'pred_high': pred_high.item(),
                'pred_low': pred_low.item()
            })
            idx_correct += 1
        else:
            examples_incorrect.append({
                'idx': idx_incorrect,
                'sentence1': s1,
                'sentence2': s2,
                'c_high': c_high,
                'c_low': c_low,
                'label_high': label_high.item(),
                'label_low': label_low.item(),
                'pred_high': pred_high.item(),
                'pred_low': pred_low.item()
            })
            idx_incorrect += 1

with open(Path(output_dir, f"{prefix}_examples_debug_correct.json"), "w") as f:
    json.dump(examples_correct, f, indent=4)

with open(Path(output_dir, f"{prefix}_examples_debug_incorrect.json"), "w") as f:
    json.dump(examples_incorrect, f, indent=4)
