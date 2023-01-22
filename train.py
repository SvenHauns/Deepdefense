import torch
import utils
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
from sklearn.metrics import roc_auc_score




def collate(batch):
    protein_list = []
    label_list = []
    info_list = []
    id_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for data in batch:
        protein_list.append(data[0])
        label_list.append(data[1])
        info_list.append(data[2])
        id_list.append(data[3])

    return [torch.tensor(protein_list, dtype=torch.float).to(device),
            torch.tensor(label_list, dtype=torch.int).to(device), torch.tensor(info_list, dtype=torch.float).to(device),
            id_list]


def train(train_loader, model, optimizer, weights, calibration_method, classes):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weighst = torch.FloatTensor(weights)
    loss_criterion = torch.nn.CrossEntropyLoss(weight=weighst.to(device))
    correct = 0
    total_loss = 0
    size = 0

    for batch in train_loader:

        model.to(device)

        batch = utils.create_encoding2(batch[0][1], batch[0][0], batch[1])
        batch = collate(batch)
        labels = batch[1]

        if labels.size()[0] == 1:
            continue

        optimizer.zero_grad()
        out = model(batch)

        if calibration_method == "DOC":

            loss = 0
            labels = labels.squeeze(dim=1).long()
            for i in range(classes):
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights[i]))
                lab = [1 if a == i else 0 for a in labels.tolist()]
                lab = torch.tensor(lab, dtype=torch.float).to(device)
                loss = loss + criterion(out[:, i], lab)

            _, out_come = out.max(dim=1)
            correct += np.equal(out_come.cpu(), labels.tolist()).sum().item()
            loss = loss * 1 / classes

        else:

            _, out_come = out.max(dim=1)

            if classes == 1:
                labels = labels.tolist()
                labels = torch.FloatTensor(labels)

            labels = labels.squeeze(dim=1).long()


            correct += out_come.squeeze().eq(labels.squeeze()).sum().item()
            labels = labels.to(device)

            loss = loss_criterion(out, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        size += len(labels)


    return total_loss / (len(train_loader)), correct / size


def validate(val_loader, model, weights, calibration_method, classes):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    size = 0

    for batch in val_loader:

        model.to(device)
        batch = utils.create_encoding2(batch[0][1], batch[0][0], batch[1])
        batch = collate(batch)
        labels = batch[1]
        if labels.size()[0] == 1:
            continue

        out = model(batch)

        if calibration_method == "DOC":

            loss = 0
            labels = labels.squeeze(dim=1).long()
            for i in range(classes):
                criterion = torch.nn.BCEWithLogitsLoss()
                lab = [1 if a == i else 0 for a in labels.tolist()]
                lab = torch.tensor(lab, dtype=torch.float).to(device)

                #out_eval = [max(min(m, 15.0), -25.0) for m in out[:, i].tolist()]
                #out_eval = torch.tensor(out_eval).to(device)

                loss = loss + criterion(out[:, i], lab)

            loss = loss * 1 / classes

            _, out_come = out.max(dim=1)
            correct += np.equal(out_come.cpu(), labels.tolist()).sum().item()

        else:

            if classes == 1:
                labels = labels.tolist()
                labels = torch.FloatTensor(labels)

            _, out_come = out.max(dim=1)

            correct += out_come.squeeze().eq(labels.squeeze()).sum().item()

            labels = labels.squeeze(dim=1).long()

            loss = loss_criterion(out, labels.to(device))

        total_loss += loss.detach()
        size += len(labels)

    return total_loss / (len(val_loader)), correct / size


def test(test_loader, models, calibration_method, classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    size = 0
    correct_class_dict = {}
    size_class_dict = {}

    model_output = []

    labels_full = []
    pred_full = []

    for batch in test_loader:

        batch = utils.create_encoding2(batch[0][1], batch[0][0], batch[1])
        batch = collate(batch)
        labels = batch[1]
        labels_full.extend(labels)

        out = []
        out_max_list = []
        for model in models:
            model.to(device)
            model.eval()
            out_model = model(batch)
            out.append(out_model.cpu().detach().numpy())

            out_pred, out_max = out_model.max(dim=1)

            out_max_list.append(out_max.tolist())

        out_max_list = np.array(out_max_list)

        out = np.array(out)
        out2 = []

        mmean = torch.tensor(np.mean(out, axis=0))
        pred_val, out_come_mean = mmean.max(dim=1)

        for enum, list_val in enumerate(out_max_list.T):
            max_val = out_come_mean.tolist()[enum]

            max_list = [mm[enum][max_val] if np.argmax(mm[enum]) == max_val else np.inf for mm in out]

            min_value = np.argmin(max_list)

            return_list = out[min_value][enum]

            out2.append(return_list)

        out = torch.tensor(out2)

        pred_full.extend(out)

        model_output.extend(out)

        if calibration_method == "DOC":

            loss = 0
            labels = labels.squeeze(dim=1).long()
            for i in range(classes):
                criterion = torch.nn.BCEWithLogitsLoss()
                lab = [1 if a == i else 0 for a in labels.tolist()]
                lab = torch.tensor(lab, dtype=torch.float)
                loss = loss + criterion(out[:, i], lab)

            loss = loss * 1 / classes
            _, out_come = out.max(dim=1)

            correct += np.equal(out_come.cpu(), labels.tolist()).sum().item()

        else:

            if classes == 1:
                labels = labels.tolist()
                labels = torch.FloatTensor(labels)

            labels = labels.squeeze(dim=1).long().cpu()
            loss = loss_criterion(out, labels)  #

            _, out_come = out.max(dim=1)

            correct += out_come.eq(labels).sum().item()

        total_loss += loss.detach() / (len(labels))
        size += len(labels)

        for class_ in range(classes):

            mask = torch.tensor([True if x == class_ else False for x in labels])
            labels_masked = torch.masked_select(labels.cpu(), mask)
            output_masked = torch.masked_select(out_come, mask)
            correct_class = output_masked.eq(labels_masked).sum().item()

            if class_ not in size_class_dict.keys():
                size_class_dict[class_] = 0
            if class_ not in correct_class_dict.keys():
                correct_class_dict[class_] = 0

            correct_class_dict[class_] = correct_class_dict[class_] + correct_class
            size_class_dict[class_] = size_class_dict[class_] + len(labels_masked)

    accuracy_by_class = []
    for key in list(correct_class_dict.keys()):
        accuracy_by_class.append(correct_class_dict[key] / max(size_class_dict[key], 1))

    total_accuracy = correct / max(1, size)

    return total_loss / (len(test_loader)), total_accuracy, accuracy_by_class, model_output
