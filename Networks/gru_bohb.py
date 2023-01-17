import torch.nn.functional as F
import torch

class gru(torch.nn.Module):
    def __init__(self, classes, arch=[41, 3, 6, 23, 0.07643942323789713]):
        super(gru, self).__init__()

        self.input_size = 1
        self.num_layers = 3
        self.bidirectional = True
        self.arch = arch
        self.arch = arch
        self.dropout = arch[4]        
        
        self.batchenorm2d = torch.nn.BatchNorm1d(28)
        self.convolution = torch.nn.Conv1d(28, arch[0], kernel_size=arch[2], stride=arch[1])

        self.GRU = torch.nn.GRU(arch[0], arch[3], bidirectional=True, dropout = self.dropout, num_layers = 2, batch_first=True)

        lin_in = 12190

        self.batch_norm1 = torch.nn.BatchNorm1d(lin_in)
        self.linear = torch.nn.Linear(lin_in, lin_in//4)
        self.batch_norm2 = torch.nn.BatchNorm1d(lin_in//4)
        self.linear2 = torch.nn.Linear(lin_in//4, lin_in//8)
        self.batch_norm3 = torch.nn.BatchNorm1d(lin_in//8)
        self.linear3 = torch.nn.Linear(lin_in//8, classes)

    def forward(self, data):

        x, z_data = data[0], data[2]
        batch_size = len(x)

        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.convolution(self.batchenorm2d(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.permute(0, 2, 1)

        hidden = torch.tensor(torch.zeros(4, batch_size, self.arch[3]).numpy())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, hidden = self.GRU(x, hidden)
        x = x.contiguous().view(batch_size, -1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.linear(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.linear2(x))
        x = self.batch_norm3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear3(x)


        return x
