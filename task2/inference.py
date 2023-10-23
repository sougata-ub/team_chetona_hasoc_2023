import pandas as pd
import torch.utils.data
from datasets import Value, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import argparse


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_list))
    model.eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    test_ds = load_dataset('json', data_files={'test': args.data_dir})['test']
    test_ds_inp = test_ds.map(lambda x: tokenizer(x['text'], max_length=256, truncation=True), batched=True).remove_columns(['text', 'id'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = torch.utils.data.DataLoader(test_ds_inp, batch_size=32, collate_fn=data_collator, shuffle=False)

    preds = []

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=masks)
            preds.extend(logits.logits.argmax(dim=-1).cpu().tolist())

    preds = [label_list[p] for p in preds]
    df = pd.DataFrame({'id': test_ds['id'], 'label': preds})
    df.to_csv(args.output_dir, index=False, header=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--task", type=str, default="binary")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    task = args.task
    if task == 'binary':
        label_list = ["HOF", "NOT"]
    elif task == 'multi':
        label_list = ['CHOF', 'NOT', 'SHOF']

    main(args)