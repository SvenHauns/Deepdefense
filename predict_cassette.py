import yaml
import argparse
import torch
import collections
import glob
import pickle
import numpy as np


    
def save_to_output(data_list, output_file, file_name):

    if data_list == []: return

    file_ = open(output_file, "a")
    
    
    for data in data_list:
    
        cassette_detected = ""
        none_counter = 0
        print(data[1][::-1])
        for enum, d in enumerate(data[1][::-1]):
            if d == "None":
                none_counter = enum
            else: break
        
        
        for cas_type in data[1][:len(data[1])-none_counter -1]:
            if cassette_detected != "": cassette_detected = cassette_detected + "_"
            cassette_detected = cassette_detected + cas_type
            
        indeces = data[2]
        
        file_.write(file_name)
        file_.write("\t")
        file_.write(str(cassette_detected))
        file_.write("\t")
        file_.write(str(indeces[0: len(indeces)-none_counter -1]))
        file_.write("\n")
        
    
    file_.close()



    return
    

def read_yaml(file_name):
    with open(file_name) as file:
        documents = yaml.full_load(file)

    return documents


def build_cassette(classes_val, cassette_dict, rules):
    confirmed_rules_dict = {}
    for enum, val in enumerate(classes_val):

        if max_val[enum] > cutoff[val]:

            current_val = val

        else:

            current_val = "None"

        if classes_pos_neg[enum] == False:
            current_val = "None"

        key_to_pop = []

        for key in cassette_dict.keys():

            current_rule_infos = cassette_dict[key]
            current_rule_num = current_rule_infos[0]
            current_indeces = current_rule_infos[1]
            none_counter = current_rule_infos[2]
            current_classes = current_rule_infos[3]
            index_values = current_rule_infos[4]

            current_rule = rules[current_rule_num]

            if current_val == "None":

                none_counter = none_counter + 1

                if none_counter > max_distance:

                    key_to_pop.append(key)

                else:
                    current_classes.append(current_val)
                    index_values.append(enum)

            else:

                if current_val in current_rule.split(" ")[current_indeces[-1] + 1:]:
                    current_index = current_indeces[-1] + 1 + current_rule.split(" ")[
                                                              current_indeces[-1] + 1:].index(current_val)

                    none_counter = 0
                    current_indeces.append(current_index)
                    current_classes.append(current_val)
                    index_values.append(enum)
                else:

                    none_counter = none_counter + 1
                    if none_counter > max_distance:
                        key_to_pop.append(key)
                    else:
                        current_classes.append("None")
                        index_values.append(enum)

            cassette_dict[key] = [current_rule_num, current_indeces, none_counter, current_classes, index_values]

        for to_pop in key_to_pop:
            cassette_dict.pop(to_pop)

        for possible_rule in rules:

            if current_val in rules[possible_rule].split(" "):
                current_rule_num = possible_rule
                current_indeces = [rules[possible_rule].split(" ").index(current_val)]
                none_counter = 0
                current_classes = [current_val]
                index_values = [enum]

                cassette_dict[str(current_rule_num) + "_" + str(enum)] = [current_rule_num, current_indeces,
                                                                          none_counter, current_classes,
                                                                          index_values]

        for key in cassette_dict.keys():

            if len([c for c in cassette_dict[key][3] if c != "None"]) >= min_correspondence:
                confirmed_rules_dict[key] = cassette_dict[key].copy()

    return confirmed_rules_dict


def extract_unique_list(confirmed_rules_dict):
    unique_list = []

    for key in confirmed_rules_dict.keys():
    

        current_rule = confirmed_rules_dict[key][3]
        current_start = key.split("_")[1]
        indeces = confirmed_rules_dict[key][4]

        current_list = [current_start, current_rule, indeces]

        if current_list not in unique_list:
            unique_list.append(current_list)

    return unique_list


def check_order(input_rule):
    key_to_pop = []
    confirmed_rules_dict = input_rule.copy()

    for en, key in enumerate(confirmed_rules_dict.keys()):

        none_counter = 0
        previous_ind = 0
        current_prots = confirmed_rules_dict[key][3]

        for val_en, value in enumerate(current_prots):

            current_rule = rules[confirmed_rules_dict[key][0]].split(" ")

            if value != "None":
                current_ind = current_rule.index(value)

                if previous_ind + none_counter + 1 >= current_ind or val_en == 0:

                    previous_ind = current_ind
                    none_counter = 0

                else:

                    key_to_pop.append(key)
                    break

            else:
                none_counter = none_counter + 1

    for to_pop in key_to_pop:
        confirmed_rules_dict.pop(to_pop)

    return confirmed_rules_dict


if __name__ == '__main__':



    cmdline_parser = argparse.ArgumentParser('cassette final prediction')

    cmdline_parser.add_argument('-t', '--data_folder',
                                default="./Dru_ype1.faa",
                                help='Name of file',
                                type=str)
    cmdline_parser.add_argument('-o', '--output_file_name',
                                default="./output.txt",
                                help='Name of output file',
                                type=str)
    cmdline_parser.add_argument('-m', '--max_distance',
                                default=2,
                                help='max distance between detected genes',
                                type=int)
    cmdline_parser.add_argument('-i', '--min_correspondence',
                                default=2,
                                help='minimum number of correspondences for a rule',
                                type=int)
                                



    args, unknowns = cmdline_parser.parse_known_args()

    dict_ = collections.defaultdict(list)

    max_distance = 2
    min_correspondence = 2

    rules = read_yaml("cassette_rules.yaml")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files = glob.glob(args.data_folder + "*classification.pkl")
    second_files = args.data_folder

    files = sorted(files)
    filter_cutoff = 0.3

    classes = ["DruA", "DruB", "DruC", "DruD", "DruE", "DruF", "DruG", "DruH", "DruM", "GajA", "GajB", "HamA",
               "HamB", "JetA", "JetB", "JetC", "JetD", "KwaA", "KwaB", "LmuA", "LmuB", "PtuA", "PtuB", "SduA", "ZorA",
               "ZorB", "ZorC", "ZorD", "ZorE", "ThsA", "ThsB"]
    pos_neg = [True, False]

    cutoff = {'DruA': 0.9038791973347255, 'DruB': 0.7121948255642072, 'DruC': 0.685823144688434,
              'DruD': 0.7667095822947629, 'DruE': 0.8732579678630655, 'DruF': 0.7776568160960042,
              'DruG': 0.7949673832920164, 'DruH': 0.5394292304039795, 'DruM': 0.6963491468307919,
              'GajA': 0.5433800663385773, 'GajB': 0.5165185936289135, 'HamA': 0.5, 'HamB': 0.6016986902881972,
              'JetA': 0.5488773294162843, 'JetB': 0.5666785275161094, 'JetC': 0.7427312126980701,
              'JetD': 0.5024555669358747, 'KwaA': 0.7721718331879318, 'KwaB': 0.5489610274356479,
              'LmuA': 0.5386696555682484, 'LmuB': 0.667394933203811, 'PtuA': 0.5123853729026722,
              'PtuB': 0.6300281324261889, 'SduA': 0.99, 'ZorA': 0.8692443040839756, 'ZorB': 0.779199939376295,
              'ZorC': 0.5772270157652536, 'ZorD': 0.5564549738181166, 'ZorE': 0.7237467068982661,
              'ThsA': 0.6048928067342549, 'ThsB': 0.5107287802712881}

    rule_count = {"Dru": 0, "Gaj": 0, "Ham": 0, "Jet": 0, "Kwa": 0, "Lmu": 0, "Ptu": 0, "Sdu": 0, "Zor": 0, "Ths": 0}

    for en, file_ in enumerate(files):

        print("##########################")
        print('current file:' + str(file_))
        print('current number: ' + str(en))
        print("##########################")

        with open(file_, 'rb') as f:
            pred_dict = pickle.load(f)

        with open(second_files + file_.split("/")[-1].split("classification.pkl")[0] + "rejection.pkl", 'rb') as f:
            pos_neg_dict = pickle.load(f)

        dict_key = list(pred_dict.keys())[0]

        argmax_val = [np.argmax([float(i) for i in l]) for l in pred_dict[dict_key]]
        max_val = [torch.sigmoid(torch.tensor(max([float(i) for i in l]))) for l in pred_dict[dict_key]]
        classes_val = [classes[a] for a in argmax_val]

        argmax_pos_neg = [np.argmax([float(i) for i in l]) for l in pos_neg_dict[dict_key]]
        max_pos_neg = [torch.sigmoid(torch.tensor(float(l[0]))) for l in pos_neg_dict[dict_key]]
        classes_pos_neg = [True if a.item() > filter_cutoff else False for a in max_pos_neg]

        cassette_dict = {}

        confirmed_rules_dict = build_cassette(classes_val, cassette_dict, rules)

        confirmed_rules_dict = check_order(confirmed_rules_dict)
        unique_list = extract_unique_list(confirmed_rules_dict)
        
        save_to_output(unique_list, args.output_file_name, file_)


        for un_list in unique_list:
            rule_count[un_list[1][0][:-1]] = rule_count[un_list[1][0][:-1]] + 1

    print("deep")
    print(rule_count)
