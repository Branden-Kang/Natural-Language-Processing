import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, D_in, D_hidden, D_out, device='cpu'):
        super(MLP, self).__init__()
        self.vocab_size = D_in
        self.D_in = D_in
        self.D_hidden = D_hidden
        self.D_out = D_out
        
        self.l1 = nn.Linear(D_in, D_hidden)
        self.l2 = nn.Linear(D_hidden, int(D_hidden/2))
        self.l3 = nn.Linear(int(D_hidden / 2), D_out)
        
        # using nn.Dropout rather than F.dropout. See https://stackoverflow.com/questions/53419474/nn-dropout-vs-f-dropout-pytorch
        self.dropout = nn.Dropout()

        self.device = device
    
    def indexTensor2sparseTensor(self, text, text_length):
        """
            Transform the document in each column in the batch to a sparse vector. Then pack them back in a sparse Tensor
            text: max_len x batch_size
            text_length: 1 x batch_size
            return: vocab_size x batch_size
        """
        values = []
        indices = []
        for col in range(text_length.size()[0]):
            for i in range(text_length[col]):
                values.append(1)
                indices.append((text[i, col], col))
        
        indices = torch.LongTensor(indices).t()
#         print (indices.size())
        values = torch.FloatTensor(values)
#         print (values.size())
        shape = (self.vocab_size, text_length.size()[0])
        
        return torch.sparse.FloatTensor(indices, values, torch.Size(shape))#.to_dense()
    
    def forward(self, text, text_length):
        x = self.indexTensor2sparseTensor(text, text_length)
        x = x.to(self.device)
        
        h = F.relu(self.l1(x.t()))
        h = self.dropout(h)
        
        h = F.relu(self.l2(h))
#         h = self.dropout(h)
        
        out = self.l3(h)
        return out