import pdb
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn import manifold
from transformers import AutoTokenizer, AutoModel

from utils import get_verbalization_ids
from data_utils import PVPS

device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(
    'roberta-large', cache_dir='pretrain/roberta-large', use_fast=False)
pvp = PVPS['mr']
model = AutoModel.from_pretrained('roberta-large',
                                  cache_dir='pretrain/roberta-large')
init_embed = model.get_input_embeddings().weight.to(device)
none_embed = torch.load(
    'output/mr/none/16-13/embeddings.pth')['word_embeddings']['weight'].to(device)
inner_embed = torch.load(
    'output/mr/inner/16-13/embeddings.pth')['word_embeddings']['weight'].to(device)
label_tokens = ['bizarre', 'memorable']

label_token_ids = [get_verbalization_ids(token, tokenizer, True)
                   for token in label_tokens]
replaced_ids = [tokenizer.vocab_size - 50 + idx
                for idx in range(len(label_tokens))]

init_vecs = init_embed[label_token_ids + replaced_ids]
none_vecs = none_embed[label_token_ids + replaced_ids]
inner_vecs = inner_embed[label_token_ids + replaced_ids]

changed_val_none, _ = (init_embed - none_embed).abs().max(dim=-1)
changed_val_inner, _ = (init_embed - inner_embed).abs().max(dim=-1)
top_changed_none = tokenizer.convert_ids_to_tokens(torch.argsort(changed_val_none, dim=-1, descending=True)[:10])
top_changed_inner = tokenizer.convert_ids_to_tokens(torch.argsort(changed_val_inner, dim=-1, descending=True)[:10])

pdb.set_trace()

"""
# selected
(Pdb) init_vecs
tensor([[-0.0831, -0.0057, -0.0385,  ...,  0.0137,  0.0196, -0.1819],
        [-0.0117, -0.0797, -0.1219,  ...,  0.0486, -0.0326, -0.1167],
        [ 0.0232, -0.0037, -0.0195,  ..., -0.0262,  0.0534,  0.0818],
        [ 0.0393,  0.0058,  0.0391,  ..., -0.0089,  0.0372, -0.0247]],
       device='cuda:0', grad_fn=<IndexBackward>)
(Pdb) none_vecs
tensor([[-0.0832, -0.0057, -0.0385,  ...,  0.0138,  0.0196, -0.1818],
        [-0.0116, -0.0797, -0.1219,  ...,  0.0486, -0.0326, -0.1168],
        [ 0.0232, -0.0037, -0.0195,  ..., -0.0262,  0.0534,  0.0818],
        [ 0.0393,  0.0058,  0.0391,  ..., -0.0089,  0.0372, -0.0247]],
       device='cuda:0')
(Pdb) inner_vecs
tensor([[-0.0833, -0.0056, -0.0385,  ...,  0.0138,  0.0197, -0.1817],
        [-0.0117, -0.0798, -0.1220,  ...,  0.0486, -0.0326, -0.1167],
        [-0.0832, -0.0057, -0.0385,  ...,  0.0138,  0.0196, -0.1818],
        [-0.0116, -0.0797, -0.1219,  ...,  0.0485, -0.0326, -0.1168]],
       device='cuda:0')

# 16-shot embeddings
(Pdb) init_vecs
tensor([[-0.0831, -0.0057, -0.0385,  ...,  0.0137,  0.0196, -0.1819],
        [-0.0117, -0.0797, -0.1219,  ...,  0.0486, -0.0326, -0.1167],
        [ 0.0232, -0.0037, -0.0195,  ..., -0.0262,  0.0534,  0.0818],
        [ 0.0393,  0.0058,  0.0391,  ..., -0.0089,  0.0372, -0.0247]],
       device='cuda:0', grad_fn=<IndexBackward>)
(Pdb) none_vecs
tensor([[-0.0831, -0.0057, -0.0385,  ...,  0.0136,  0.0197, -0.1820],
        [-0.0117, -0.0798, -0.1219,  ...,  0.0487, -0.0326, -0.1166],
        [ 0.0232, -0.0037, -0.0195,  ..., -0.0262,  0.0534,  0.0818],
        [ 0.0393,  0.0058,  0.0391,  ..., -0.0089,  0.0372, -0.0247]],
       device='cuda:0')
(Pdb) inner_vecs
tensor([[-0.0831, -0.0057, -0.0385,  ...,  0.0137,  0.0196, -0.1819],
        [-0.0117, -0.0797, -0.1220,  ...,  0.0487, -0.0327, -0.1168],
        [-0.0831, -0.0057, -0.0385,  ...,  0.0136,  0.0197, -0.1820],
        [-0.0117, -0.0798, -0.1219,  ...,  0.0487, -0.0326, -0.1166]],
       device='cuda:0')
"""
