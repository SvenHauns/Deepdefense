"""

credit: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

"""

import torch
from torch import nn, optim
import utils
import train
import random


class ModelWithTemperature(nn.Module):

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1)

    def forward(self, input):
        logits = self.model(input)
        a = self.temperature_scale(logits)

        return a

    def temperature_scale(self, logits):

        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))

        return logits / temperature

    def set_temperature2(self, temp):

        self.temperature = temp
        
        return self

    def set_temperature(self, valid_loader, classes):
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in valid_loader:
            
                batch = utils.create_encoding2(batch[0][1], batch[0][0], batch[1])
                batch = train.collate(batch)

                logits = self.model(batch)
                _, out_come = logits.max(dim=1)

                labels = [random.randint(0, classes) for _ in out_come.tolist()]
                labels = torch.tensor([out if out < classes else 0 for out in labels]).long()

                labels_list.append(labels)
                logits_list.append(logits)

            for logit in logits_list[1:-1]:
                torch.stack([logits_list[0], logit], axis=0)

            for lab in labels_list[1:-1]:
                torch.stack([labels_list[0], lab], axis=0)

            logits = logits_list[0]
            labels = labels_list[0]

        before_temperature_nll = nll_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        optimizer = optim.LBFGS([self.temperature], lr=0.00001, max_iter=100)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self, self.temperature
