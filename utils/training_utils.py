import numpy as np

from sklearn.metrics import f1_score

import torch
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm import trange
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def train_by_trainer_api(model, tokenizer, train_dataset, val_dataset, epochs, device):

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        learning_rate=4e-4
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=128)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        data_collator=data_collator
    )

    trainer.train()



def train(model, train_dataloader, optimizer, validation_dataloader, epochs, device):
    for _ in trange(epochs, desc = 'Epoch'):    
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        loss_fn = torch.nn.CrossEntropyLoss()

        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            train_output = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask)
            logits = train_output.logits
            preds = torch.softmax(logits, dim=-1)
            loss = loss_fn(preds, b_labels)

            loss.backward()
            optimizer.step()
            tr_loss += loss
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        model.eval()
        nb_batches = 0
        f1 = 0
        for step, batch in tqdm(enumerate(validation_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                eval_output = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask)
            b_labels = b_labels.cpu().detach().numpy()
            logits = eval_output.logits.cpu().detach().numpy()
            preds = np.argmax(logits, axis=1)
            f1 += f1_score(b_labels, preds, average='macro')
            nb_batches+=1
        print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        print('\t - Validation F1: {:.4f}\n'.format(f1/nb_batches))

def eval(model, validation_dataloader, device):
    model.eval()
    nb_batches = 0
    f1 = 0
    for step, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            eval_output = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask)
        b_labels = b_labels.cpu().detach().numpy()
        logits = eval_output.logits.cpu().detach().numpy()
        preds = np.argmax(logits, axis=1)
        f1 += f1_score(b_labels, preds, average='macro')
        nb_batches+=1
    print('\t - Validation F1: {:.4f}\n'.format(f1/nb_batches))

def predict(tweets, model, tokenizer, device):
    eval_ids = []
    eval_attention_mask = []
    def preprocessing(input_text, tokenizer):
        return tokenizer.encode_plus(input_text, add_special_tokens = True, max_length = 256, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    for sample in tqdm(tweets):
        encoding_dict = preprocessing(sample, tokenizer)
        eval_ids.append(encoding_dict['input_ids']) 
        eval_attention_mask.append(encoding_dict['attention_mask'])
    eval_ids = torch.cat(eval_ids, dim = 0)
    eval_attention_mask = torch.cat(eval_attention_mask, dim = 0)
    eval_set = TensorDataset(eval_ids, eval_attention_mask)
    eval_dataloader = DataLoader(eval_set, sampler = SequentialSampler(eval_set), batch_size = 256, shuffle=False)
    predictions = np.array([])
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            eval_output = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask)
        logits = eval_output.logits.cpu().detach().numpy()
        preds = np.argmax(logits, axis=1)
        predictions = np.append(predictions, preds, axis=0)
    return predictions