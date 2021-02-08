# coding: utf-8
import argparse
import time
import math
import os
import sys
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import data
import model_1
import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 FNN Language Model')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='FNN',
                    help='type of recurrent net (FNN)')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.04,
#parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='initial momentum')
parser.add_argument('--weight_decay', type=float, default=1e-8,
                    help='weight_decay')
parser.add_argument('--max_grad_norm', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--window_size', type=int, default=13,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='fnnmodel.pt',
                    help='path to save the final model')
parser.add_argument('--onnx_export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--dry_run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed_all(seed)
# Set the random seed manually for reproducibility.
set_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
print(torch.cuda.device_count())
#torch.cuda.set_device(7)
device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    #data = data.view(bsz, -1).t().contiguous()
    data = data.view(bsz, -1)
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_1.FNNModel(args.model, ntokens, args.embedding_size, args.window_size, args.hidden_size, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(), args.lr)
optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)
#optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, args.weight_decay)
#optimizer = optim.Adadelta(model.parameters())
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.window_size, source.size(1) - 1 - i)
    data = source[:,i:i+seq_len]
    target = source[:,i+seq_len:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(1) - args.window_size):
            data, targets = get_batch(data_source, i)
            output = model(data)
            total_loss += criterion(output, targets).item()
    return total_loss / (data_source.size(1) - args.window_size)


def train():
    # Turn on training mode which enables dropout.
    model.train()

    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(1) - args.window_size)):
        data, targets = get_batch(train_data, i)
        #print(data.shape)
        #print(targets.shape)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        #for p in model.parameters():
        #    p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        #print(total_loss)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, train_data.size(1) - 1 - args.window_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break



# Loop over epochs.
lr = args.lr
best_val_loss = None


# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        sys.stdout.flush()
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            print("good than previous best!!!!!!!!!!!!!!!!")
            print("do one test!!!!!")
            test_loss = evaluate(test_data)
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
            print('=' * 89)
            sys.stdout.flush()
        else:
            print("not good than previous best..........")
            print("still do one test!!!!!")
            test_loss = evaluate(test_data)
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
            print('=' * 89)
            sys.stdout.flush()
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #lr /= 4.0
            for p in optimizer.param_groups:
                p['lr'] *= 0.50
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
sys.stdout.flush()

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
        format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.window_size)
