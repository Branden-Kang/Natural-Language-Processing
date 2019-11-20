import pandas as pd
import torch
from torch import nn
from model import MLP
from preprocessing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_to_split = './data'
model_path = './model'

train_iter, valid_iter, test_iter, vocab = read_test_valid(path_to_split, device = device)

HIDDEN_SIZE = 128
num_classes = 2
model = MLP(len(vocab), HIDDEN_SIZE, num_classes, device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-5)
loss_func = nn.CrossEntropyLoss(reduction = 'mean')

model.to(device)
loss_func.to(device)

save_path = './best_model.pt'

def train_validate(num_epoches = 2, save_path=model_path):
    
    # I don't how many epoches are enough, so we will track the best performing model
    best_acc = 0
    best_epoch = -1
    to_save = {}
    
    for e in range(num_epoches):
        # because of dropout layer, we turn training on
        model.train()
        epoch_loss = 0
        for x, y in train_iter:
            
            optimizer.zero_grad()

            text, text_lengths = x
#             print(text.device)
#             print(text_lengths.device)
#             print(y.device)
#             break
            out = model(text, text_lengths)
            
            loss = loss_func(out, y)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print('epoch = {}, loss = {}'.format(e, epoch_loss / len(train_iter)))
        correct, total = 0, 0
        
        
        # enter evaluation mode, no need for grads to make it run faster
        with torch.no_grad():
            # because of dropout layer, we turn training off
            model.eval()
            for val_x, val_y in valid_iter:
                text, text_lengths = val_x
    #             print(text.device)
    #             print(text_lengths.device)
    #             print(val_y.device)
                # dim of out is batch_size x num_classes
                out = model(text, text_lengths)
                correct += torch.max(out, 1)[1].eq(val_y).sum()
                total += out.size()[0]
            cur_acc = float(correct) / total
            print('Validation accuracy = {}'.format(cur_acc))
            
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_epoch = e
                
                # record the model state_dict() for saving later
                to_save = {
                    'epoch': e,
                    'model_state_dict': model.state_dict()
                }
                
    # report and save the best model
    print(f'best epoch = {best_epoch} with best validation accuracy = {best_acc}')
    torch.save(to_save, save_path + '/best_model.pt')

def load_model(save_path):
      # torch.save(to_save, save_path)
      model = MLP(len(vocab), HIDDEN_SIZE, num_classes, device=device)

      checkpoint = torch.load(save_path + '/best_model.pt')
      model.load_state_dict(checkpoint['model_state_dict'])
      epoch = checkpoint['epoch']

      # move the model to GPU if has one
      model.to(device)

      # need this for dropout
      model.eval()
      return model


def test_model(model):
      correct, total = 0, 0
      with torch.no_grad():
          # because of dropout layer, we turn training off
          model.eval()
          for test_x, test_y in test_iter:
              text, text_lengths = test_x
              # dim of out is batch_size x num_classes
              out = model(text, text_lengths)
              correct += torch.max(out, 1)[1].eq(test_y).sum()
              total += out.size()[0]
          cur_acc = float(correct) / total
          print('Test accuracy = {}'.format(cur_acc))

train_validate()
model = load_model(model_path)
test_model(model)