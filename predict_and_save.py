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
import subprocess as sp

functions_nn = {'gru2_bohb': Networks.gru2_bohb.gru,
                'gru_bohb': Networks.gru_bohb.gru}
                
                
def prodigal(prodigal_cmd, fasta_file, completeness):
    meta = ' -p meta ' if completeness == 'partial' else ''
    fasta_file_preffix = fasta_file.rsplit('.', 1)[0]
    output_fasta_file = fasta_file_preffix + '_proteins.fa'
    log_file = fasta_file_preffix + '_prodigal.log'
    prodigal_cmd += ' -i {input_fasta}  -c -m -g 11 -a {output_fasta} -q' + meta
    
    print("fasta_file")
    print(fasta_file)
    
    prodigal_cmd = prodigal_cmd.format(prodigal=prodigal_cmd, input_fasta=fasta_file, output_fasta=output_fasta_file)

    with open(log_file, 'w') as lf:
        sp.call(prodigal_cmd.split(), stdout=lf)
        
    file_reformat(output_fasta_file)

    return output_fasta_file
    
    
def file_reformat(file_):

    
    lines_read = open(file_).readlines()
    file_write = open(file_, "w")
    	
    for en, line_ in enumerate(lines_read):
    	
    	
    	if line_[0] == ">":
    		if en != 0:file_write.write("\n")
    		file_write.write(lines_read[en])
    		
    	else:
    		
    		seq = ""

    		for char in line_:

    			if char == "*" or char == "-" or char == "\n":continue
    			seq = seq + char
    		
    		file_write.write(seq)

    	
    file_write.close()
    return
    	




def main_func(args, dict_, file_, model_list_ens, classes, calibration_method = "DOC"):
    max_lines = len(open(file_).readlines())
    if max_lines <= 1:
        return dict_

    dataset_test = database.MyOwnDatasetClass([[file_]]).return_data()
    test_loader = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)



    _, _, _, model_output = train.test(test_loader, model_list_ens, calibration_method, classes)

    model_output = [m.tolist() for m in model_output]

    dict_[file_] = model_output

    return dict_


# samtools view ?


if __name__ == '__main__':



    cmdline_parser = argparse.ArgumentParser('cassette final prediction')

    cmdline_parser.add_argument('-t', '--data_folder',
                                default="./data/",
                                help='folder for the input data',
                                type=str)
    cmdline_parser.add_argument('-c', '--prodigal_completeness',
                                default="partial",
                                help='completeness',
                                type=str)
    cmdline_parser.add_argument('-p', '--prodigal',
                                default=False,
                                help='use prodigal to create proteins',
                                type=bool)
    cmdline_parser.add_argument('-r', '--result_folder',
                                default="./results/",
                                help='folder for the output data',
                                type=str)
                                

    args, unknowns = cmdline_parser.parse_known_args()

    dict_ = collections.defaultdict(list)
    dictionary_input = utils.read_yaml()
    files = glob.glob(args.data_folder + "*")
    files = sorted(files)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    ######## classification models #########

    classes = 31
    path_to_models = ["./models/models_classification/model_gru2_bohb_unscaled_0_.pt", "./models/models_classification/model_gru2_bohb_unscaled_1_.pt",
                      "./models/models_classification/model_gru2_bohb_unscaled_2_.pt"]

    temperature = [1.6040, 1.6209, 1.4967]
    model_list_ens = []
    for model_num, model_name in enumerate(path_to_models):
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        model = functions_nn[checkpoint["model_type"]](classes).to(device)
        model.load_state_dict(checkpoint["model"])
        scaled_model = calibration.temp_scaling.ModelWithTemperature(model)
        scaled_model = scaled_model.set_temperature2(torch.nn.Parameter(torch.tensor([temperature[model_num]])))
        model_list_ens.append(scaled_model)
        
        
        
    ######## rejection model #########
        

    classes = 2
    path_to_models = ["./models/models_rejection/model_gru2_bohb_unscaled_0_.pt", "./models/models_rejection/model_gru2_bohb_unscaled_1_.pt",
                      "./models/models_rejection/model_gru2_bohb_unscaled_2_.pt"]
    model_list_rejection_ens = []
    for model_num, model_name in enumerate(path_to_models):
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        model = functions_nn[checkpoint["model_type"]](classes).to(device)
        model.load_state_dict(checkpoint["model"])
        model_list_rejection_ens.append(model)
        

    for en, file_ in enumerate(files):

        print("##########################")
        print('current file:' + str(file_))
        print('current number: ' + str(en))
        print("##########################")
        
        
        if args.prodigal == True:
            file_ = prodigal("prodigal", file_, args.prodigal_completeness)
            

        classes = 31
        dict_ = main_func(args, dict_, file_, model_list_ens, classes)

        if dict_[file_] == []:
            dict_ = collections.defaultdict(list)
            continue

        with open(args.result_folder + str(file_.split("/")[-1].split(".fa")[0]) + '_classification.pkl', 'wb') as handle:
            pickle.dump(dict_, handle)
        dict_ = collections.defaultdict(list)
        
        classes = 2
        dict_ = main_func(args, dict_, file_, model_list_rejection_ens, classes)

        if dict_[file_] == []:
            dict_ = collections.defaultdict(list)
            continue

        with open(args.result_folder + str(file_.split("/")[-1].split(".fa")[0]) + '_rejection.pkl', 'wb') as handle:
            pickle.dump(dict_, handle)
        dict_ = collections.defaultdict(list)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
