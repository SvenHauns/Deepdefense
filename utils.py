from Bio import SeqIO
import numpy as np
import yaml
import get_protparam_parser
from sklearn.utils import class_weight


def load_protein_data(file_list):
    proteins = []

    for file_ in file_list:
        for record in SeqIO.parse(file_, "fasta"):
            proteins.append((str(record.id), str(record.seq)))

    return proteins


"""
The function read_yaml  reads in the used aminoacids
"""


def read_yaml():
    with open(r'./aminoacids.yaml') as file:
        documents = yaml.full_load(file)

    return documents


def create_encoding2(proteins, protein_ids, protein_type, maximum_input_length=800):
    dic_amino = read_yaml()
    data_list = []

    for protein_num, protein in enumerate(proteins):

        onehot_encoded = []
        size = len(dic_amino.keys()) + 2

        for char in protein:

            if char not in dic_amino.keys():
                val = size
            else:
                val = dic_amino.get(char) + 1

            listofzeros = [0] * size
            listofzeros[val - 1] = 1
            onehot_encoded.append(listofzeros)

        listofzeros = [0] * size
        num = maximum_input_length - len(onehot_encoded)
        for g in range(num):
            onehot_encoded.append(listofzeros)

        onehot_encoded = onehot_encoded[:maximum_input_length]
        x = onehot_encoded
        y = [protein_type[protein_num]]
        additional_info = get_protparam_parser.parser(protein)
        z = additional_info
        data = [x, y, z, str(protein_ids[protein_num])]
        data_list.append(data)

    return data_list


"""
The function calculate_weight_vector calculates weight vector for training
arguments:
train_data: train data to be weighted
"""


def calculate_weight_vector(train_data):

    data_y = [data[1] for data in train_data]
    weight_vector = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(data_y),
                                                      y=np.array(data_y))

    print("The weight vector is:")
    print(list(weight_vector))

    return list(weight_vector)
