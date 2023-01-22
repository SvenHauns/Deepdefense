
import torch.nn.functional as F
import torch


class gru(torch.nn.Module):
    def __init__(self, classes, arch=[45, 7, 5, 20, 0.23128823035454756, 71]):
        super(gru, self).__init__()
        
        self.input_size = 1
        self.num_layers = 3
        self.bidirectional = True
        self.arch = arch
        self.arch = arch
        hidden_size = arch[-1]
        self.dropout = arch[4]
        
        self.batchenorm2d = torch.nn.BatchNorm1d(28)
        self.convolution = torch.nn.Conv1d(28, arch[0], kernel_size=arch[1], stride=arch[2])
        
        out = int(((800 - int(7))/5)+1)

        self.GRU = torch.nn.GRU(arch[0], arch[3], bidirectional=True, dropout=self.dropout, num_layers = 2, batch_first=True)

        lin_in = arch[3]*2 * out + hidden_size

        self.batch_norm1 = torch.nn.BatchNorm1d(lin_in)
        self.linear = torch.nn.Linear(lin_in, lin_in//4)
        self.batch_norm3 = torch.nn.BatchNorm1d(lin_in//4)
        self.linear3 = torch.nn.Linear(lin_in//4, lin_in//8)
        self.batch_norm4 = torch.nn.BatchNorm1d(lin_in//8)
        self.linear4 = torch.nn.Linear(lin_in//8, classes)
        
        self.batch_norm1_add = torch.nn.BatchNorm1d(12)
        self.linear1_add = torch.nn.Linear(12, hidden_size)
        self.batch_norm2_add = torch.nn.BatchNorm1d(hidden_size)
        self.linear2_add = torch.nn.Linear(hidden_size, hidden_size)
        self.batch_norm3_add = torch.nn.BatchNorm1d(hidden_size)
        self.linear3_add = torch.nn.Linear(hidden_size, hidden_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def forward(self, data):

        x, z_data = data[0], data[2]
        batch_size = len(x)

        z_data = F.leaky_relu(self.linear1_add(self.batch_norm1_add(z_data)))
        z_data = F.leaky_relu(self.linear2_add(self.batch_norm2_add(z_data)))
        z_data = F.leaky_relu(self.linear3_add(self.batch_norm3_add(z_data)))

        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.convolution(self.batchenorm2d(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.permute(0, 2, 1)

        hidden = torch.tensor(torch.zeros(4, batch_size, self.arch[3]).numpy()).to(self.device)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, hidden = self.GRU(x, hidden)

        x = x.contiguous().view(batch_size, -1)
        x = torch.cat([x, z_data], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.batch_norm1(x)
        x = F.leaky_relu(self.linear(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.batch_norm3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.linear3(x))
        x = self.batch_norm4(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear4(x)

        return x
