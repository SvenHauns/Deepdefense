import utils
import torch
from torch.utils.data import DataLoader
import database
import glob
import argparse
import Networks
import collections
import train
import pickle
import calibration
import yaml

functions_nn = {'gru2_bohb': Networks.gru2_bohb.gru,
                'gru_bohb': Networks.gru_bohb.gru}


def read_yaml(file_name):
    with open(file_name) as file:
        documents = yaml.full_load(file)

    return documents


def main_func(args, dict_, file_, model_list_ens):
    max_lines = len(open(file_).readlines())
    if max_lines <= 1:
        return dict_

    dataset_test = database.MyOwnDatasetClass([[file_]]).return_data()
    test_loader = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)

    calibration_method = "DOC"
    classes = 2

    _, _, _, model_output = train.test(test_loader, model_list_ens, calibration_method, classes)

    model_output = [m.tolist() for m in model_output]

    dict_[file_] = model_output

    return dict_


# samtools view ?


if __name__ == '__main__':

    def str_to_bool(in_string):

        if in_string == "True" or in_string == "true" or in_string == "TRUE":
            return True

        else:
            return False


    cmdline_parser = argparse.ArgumentParser('cassette final prediction')

    cmdline_parser.add_argument('-t', '--file_name',
                                default="./Dru_ype1.faa",
                                help='Name of file',
                                type=str)
    cmdline_parser.add_argument('-c', '--completenes',
                                default="partial",
                                help='completenes',
                                type=str)
    cmdline_parser.add_argument('-z', '--add_infos',
                                default="True",
                                help='Additional information for classification',
                                action='store_true')
    cmdline_parser.add_argument('-m', '--load_genes',
                                default="True",
                                help='load genes or proteins',
                                type=str)
    cmdline_parser.add_argument('-s', '--start_value',
                                default=0,
                                help='start value for reading a file',
                                type=int)
    cmdline_parser.add_argument('-a', '--calculate_infos',
                                default="True",
                                help='start value for reading a file',
                                action='store_true')
    cmdline_parser.add_argument('-x', '--base_class',
                                default=0,
                                help='start value for reading a file',
                                type=int)
    cmdline_parser.add_argument('-j', '--sub_class',
                                default=0,
                                help='start value for reading a file',
                                type=int)
    cmdline_parser.add_argument('-g', '--files_restirction',
                                default=0,
                                help='files to read from',
                                type=int)

    args, unknowns = cmdline_parser.parse_known_args()

    dict_ = collections.defaultdict(list)
    dictionary_input = utils.read_yaml()
    model_path = "./files/"

    files = glob.glob("./data/" + "*.fa")
    files = sorted(files)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    length_ = len(files)
    exception_counter = 0

    classes = 2
    path_to_models = ["/models/model_gru_bohb_unscaled_0_.pt", "./models/model_gru_bohb_unscaled_1_.pt",
                      "./models/model_gru_bohb_unscaled_2_.pt"]

    temperature = [1.6040, 1.6209, 1.4967]

    model_list_ens = []
    for model_num, model_name in enumerate(path_to_models):
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        model = functions_nn[checkpoint["model_type"]](classes).to(device)
        model.load_state_dict(checkpoint["model"])
        scaled_model = calibration.temp_scaling.ModelWithTemperature(model)
        scaled_model = scaled_model.set_temperature2(torch.nn.Parameter(torch.tensor([temperature[model_num]])))
        model_list_ens.append(scaled_model)

    for en, file_ in enumerate(files):

        print("##########################")
        print('current file:' + str(file_))
        print('current number: ' + str(en))
        print("##########################")

        dict_ = main_func(args, dict_, file_, model_list_ens)

        if dict_[file_] == []:
            dict_ = collections.defaultdict(list)
            continue

        with open("./result/" + str(file_.split("/")[-1].split(".fa")[0]) + '_algo.pkl', 'wb') as handle:
            pickle.dump(dict_, handle)
        dict_ = collections.defaultdict(list)
