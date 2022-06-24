import os
import random

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizer import get_tokenizer_wordlevel, get_tokenizer_bpe

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

def pad_collate_fn(batch):
    src_sentences,trg_sentences=[],[]
    for sample in batch:
        src_sentences+=[sample[0]]
        trg_sentences+=[sample[1]]

    src_sentences = pad_sequence(src_sentences, batch_first=True, padding_value=0)
    trg_sentences = pad_sequence(trg_sentences, batch_first=True, padding_value=0)

    return src_sentences, trg_sentences

class TranslationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_encoded=self.dataset[idx]['translation_src']
        trg_encoded=self.dataset[idx]['translation_trg']
        
        return (
            torch.tensor(src_encoded),
            torch.tensor(trg_encoded),
        )

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):

        # Dataset is already sorted so just chunk indices
        # into batches of indices for sampling
        self.batch_size=batch_size
        self.indices=range(len(dataset))
        self.batch_of_indices=list(chunk(self.indices, self.batch_size))
        self.batch_of_indices = [batch.tolist() for batch in self.batch_of_indices]
    
    def __iter__(self):
        random.shuffle(self.batch_of_indices)
        return iter(self.batch_of_indices)
    
    def __len__(self):
        return len(self.batch_of_indices)


def get_data(example_cnt):
    data_train=load_dataset('wmt16','tr-en',split='train')
    data_val=load_dataset('wmt16','tr-en',split='validation')
    #data=load_dataset('wmt16','tr-en',split='test')
    #data=data.select(range(example_cnt)) 
    data_train=data_train.flatten()
    data_val=data_val.flatten()
    data_train=data_train.rename_column('translation.en','translation_trg')
    data_val=data_val.rename_column('translation.en','translation_trg')
    data_train=data_train.rename_column('translation.tr','translation_src')
    data_val=data_val.rename_column('translation.tr','translation_src')
    #print(type(data))

    return data_train,data_val

def preprocess_data(data, tokenizer, max_seq_len, test_proportion):

    # Tokenize
    def tokenize(example):
        return {
            'translation_src': tokenizer.encode(example['translation_src']).ids,
            'translation_trg': tokenizer.encode(example['translation_trg']).ids,
        }
    data=data.map(tokenize)

    # Compute sequence lengths
    def sequence_length(example):
        return {
            'length_src': [len(item) for item in example['translation_src']],
            'length_trg': [len(item) for item in example['translation_trg']],
        }
    data=data.map(sequence_length, batched=True, batch_size=10000)

    # Filter by sequence lengths
    def filter_long(example):
        return example['length_src']<= max_seq_len and example['length_trg']<=max_seq_len
    data=data.filter(filter_long)

    # Split 
    #data=data.train_test_split(test_size=test_proportion)

    # Sort each split by length for dynamic batching (see CustomBatchSampler)
    data=data.sort('length_src', reverse=True)
    #data['test']=data['test'].sort('length_src', reverse=True)

    return data


def get_translation_dataloaders(
    dataset_size,
    vocab_size,
    tokenizer_type,
    tokenizer_save_pth,
    test_proportion,
    batch_size,
    max_seq_len,
    report_summary,
    ):

    data_train,data_val=get_data(dataset_size)
 
    if tokenizer_type == 'wordlevel':
        tokenizer=get_tokenizer_wordlevel(data_train, vocab_size)
    elif tokenizer_type == 'bpe':
        tokenizer=get_tokenizer_bpe(data_train, vocab_size)

    # Save tokenizers
    tokenizer.save(tokenizer_save_pth)

    data_train=preprocess_data(data_train, tokenizer, max_seq_len, test_proportion)
    data_val=preprocess_data(data_val, tokenizer, max_seq_len, test_proportion)

    if report_summary:
        wandb.run.summary['train_len']=len(data_train)
        wandb.run.summary['val_len']=len(data_val)

    # Create pytorch datasets
    train_ds=TranslationDataset(data_train)
    val_ds=TranslationDataset(data_val)

    #print(type(train_ds))
    #print(type(val_ds))

    # Create a custom batch sampler
    custom_batcher_train = CustomBatchSampler(train_ds, batch_size)
    custom_batcher_val= CustomBatchSampler(val_ds, batch_size)

    # Create pytorch dataloaders
    train_dl=DataLoader(train_ds, collate_fn=pad_collate_fn, batch_sampler=custom_batcher_train, pin_memory=True)
    val_dl=DataLoader(val_ds, collate_fn=pad_collate_fn, batch_sampler=custom_batcher_val, pin_memory=True)

    return train_dl, val_dl
from transformers import PreTrainedTokenizerFast
def get_test_data():
    data_test=load_dataset('wmt16','tr-en',split='test')
    data_test=data_test.flatten()
    data_test=data_test.rename_column('translation.en','translation_trg')
    data_test=data_test.rename_column('translation.tr','translation_src')
    return data_test

def preprocess_data2(data, tokenizer, max_seq_len):

    # Tokenize
    def tokenize2(example):
        return {
            'translation_src': tokenizer.encode(example['translation_src']),
            'translation_trg': tokenizer.encode(example['translation_trg']),
        }
    data=data.map(tokenize2)

    # Compute sequence lengths
    def sequence_length2(example):
        return {
            'length_src': [len(item) for item in example['translation_src']],
            'length_trg': [len(item) for item in example['translation_trg']],
        }
    data=data.map(sequence_length2, batched=True, batch_size=10000)

    # Filter by sequence lengths
    def filter_long(example):
        return example['length_src']<= max_seq_len and example['length_trg']<=max_seq_len
    data=data.filter(filter_long)

    # Split 
    #data=data.train_test_split(test_size=test_proportion)

    # Sort each split by length for dynamic batching (see CustomBatchSampler)
    data=data.sort('length_src', reverse=True)
    #data['test']=data['test'].sort('length_src', reverse=True)

    return data

"""Decided not to use this, using upper method to get data directly instead
from tokenizers import Tokenizer

def getTestDataset():
    data_test =get_test_data()
 
    tok_path = "../runs/TR-ENG Translation Test using Train-Val default datasets/tokenizer.json"
    #tokenizer = PreTrainedTokenizerFast(tokenizer_file = tok_path)
    tokenizer = Tokenizer.from_file(tok_path)
    #data_test=preprocess_data2(data_test, tokenizer, 40)
    data_test=preprocess_data(data_test, tokenizer, 40,None)

    test_ds=TranslationDataset(data_test)

    return test_ds
"""
