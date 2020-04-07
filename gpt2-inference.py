import os
import json
import glob
import time
    
import mxnet as mx
import gluonnlp as nlp

from gluonnlp.data import SentencepieceTokenizer
from kogpt2.model.gpt import GPT2Model as MXGPT2Model
from kogpt2.utils import get_tokenizer

def get_kogpt2_model(model_file,
                     vocab_file,
                     ctx=mx.cpu(0)):
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    mxmodel = MXGPT2Model(units=768,
                          max_length=1024,
                          num_heads=12,
                          num_layers=12,
                          dropout=0.1,
                          vocab_size=len(vocab_b_obj))
    mxmodel.load_parameters(model_file, ctx=ctx)
    
    return (mxmodel, vocab_b_obj)

def model_fn(model_dir):    
    voc_file_name = glob.glob('{}/*.spiece'.format(model_dir))[0]
    model_param_file_name = glob.glob('{}/*.params'.format(model_dir))[0]
    
    # check if GPU is available
    if mx.context.num_gpus() > 0:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
        
    model, vocab = get_kogpt2_model(model_param_file_name, voc_file_name, ctx)
    tok = SentencepieceTokenizer(voc_file_name)
    
    return model, vocab, tok, ctx

def transform_fn(model, request_body, content_type, accept_type):
    model, vocab, tok, ctx = model
    
    sent = request_body.encode('utf-8')
    sent = sent.decode('unicode_escape')[1:]
    sent = sent[:-1]
    toked = tok(sent)
    
    t0 = time.time()
    inference_count = 0
    while 1:
        input_ids = mx.nd.array([vocab[vocab.bos_token]]  + vocab[toked]).expand_dims(axis=0)
        pred = model(input_ids.as_in_context(ctx))[0]
        gen = vocab.to_tokens(mx.nd.argmax(pred, axis=-1).squeeze().astype('int').asnumpy().tolist())[-1]
        if gen == '</s>':
            break
        sent += gen.replace('▁', ' ')
        toked = tok(sent)
        inference_count += 1
    
    response_body = json.dumps([sent, inference_count, time.time() - t0])
    
    return response_body, content_type