import pandas as pd
import torch
import torchtext
from torchtext.data import Field, LabelField
from torchtext import data, datasets
from torchtext.data import TabularDataset, BucketIterator, Iterator

def split_save_data(path_to_csv, path_to_split):
	# path = './reviews_Cell_Phones_and_Accessories_5.csv'
	corpus = pd.read_csv(path_to_csv)

	# create small datasets for quick tweaking
	train_small = corpus[:16000]
	valid_small = corpus[16000:20000]
	test_small = corpus[20000:25000]
	# write to files
	train_small.to_csv(path_to_split + '/train_small.csv', index=False)
	valid_small.to_csv(path_to_split + '/valid_small.csv', index=False)
	test_small.to_csv(path_to_split + '/test_small.csv', index=False)

class BatchWrapper:
    def __init__(self, iterator, x_var, y_var):
        self.iterator, self.x_var, self.y_var = iterator, x_var, y_var # we pass in the list of attributes for x 
        print (self.y_var)

    def __iter__(self):
        for batch in self.iterator:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            y = getattr(batch, self.y_var)
            y[y < 4] = 0
            y[y >= 4] = 1
            yield (x, y)

    def __len__(self):
        return len(self.iterator)

def read_test_valid(path_to_split, device = 'cpu'):

	BATCH_SIZE = 64
	MAX_VOCAB_SIZE = 25_000

	# a very simple tokenizer
	tokenize = lambda x:x.split()

	# Fields from torchtext: specifying how to process each field in the CSV files

	TEXT = Field(sequential=True,
             tokenize=tokenize,
             lower=True,
             include_lengths=True)

	LABEL = Field(sequential=False,
	             use_vocab=False,
	             dtype = torch.long)

	fields = [('reviewText', TEXT), ('overall', LABEL)]

	# load train, validation, and test data all in once
	train_data, valid_data, test_data = TabularDataset.splits(
	            path='',
	            train=path_to_split + '/train_small.csv',
	            validation= path_to_split + '/valid_small.csv',
	            test = path_to_split + '/test_small.csv',
	            format='csv',
	            skip_header=True,
	            fields=fields)

	# the vocab can only be built from the training portion
	TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
	LABEL.build_vocab(train_data)
	LABEL.build_vocab(train_data)
	vocab = TEXT.vocab

	train_iter = BucketIterator(
	    train_data,
	    batch_size = BATCH_SIZE,
	    device = device,
	    sort_key=lambda x: len(x.reviewText),
	    sort_within_batch=False,
	    repeat=False
	)

	valid_iter = BucketIterator(
	    valid_data,
	    batch_size = BATCH_SIZE * 4,
	    device = device,
	    repeat = False
	)

	test_iter = BucketIterator(
        test_data,
        batch_size = BATCH_SIZE * 4,
        device = device,
        repeat = False
      )
      
	train_iter = BatchWrapper(train_iter, 'reviewText', 'overall')
	valid_iter = BatchWrapper(valid_iter, 'reviewText', 'overall')
	test_iter = BatchWrapper(test_iter, 'reviewText', 'overall')

	return train_iter, valid_iter, test_iter, vocab