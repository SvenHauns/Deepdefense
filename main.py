import utils
import train
import torch
from torch.utils.data import DataLoader
import math
import database
import load_save
import random
import argparse
import Networks

functions_nn = {'gru2_bohb': Networks.gru2_bohb.gru,
                'gru_bohb': Networks.gru_bohb.gru}


def init_all(model):
    for p in model.parameters():
        torch.manual_seed(random.randint(0, 10000))
        init_func = init_funcs.get(len(p.shape), init_funcs[2])
        init_func(p)


init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.),
    2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.),  
    3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.),  
    4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), 
    "default": lambda x: torch.nn.init.constant(x, 1.), 
}


def run_net(train_loader, val_loader, weights, saver, network_type, epochs, calibration_method, classes,
            ensemble_num=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = functions_nn[network_type](classes).to(device)
    init_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 12, 20, 30, 40, 50, 60, 70, 90, 100],
                                                    gamma=0.9, last_epoch=-1)

    best_validation = - math.inf
    best_epoch = 0
    best_epoch_acc = 0
    saved_model = type(model)(classes)

    not_changed_counter = 0

    for epoch in range(epochs):

        loss, train_acc = train.train(train_loader, model, optimizer, weights, calibration_method, classes)
        validation, val_acc = train.validate(val_loader, model, weights, calibration_method, classes)

        
        if epoch > epochs / 3:
            not_changed_counter = not_changed_counter + 1

        if val_acc > best_validation and epoch > epochs / 3:
            not_changed_counter = 0
            best_validation = val_acc
            saved_model.load_state_dict(model.state_dict())
            best_epoch = epoch

            saver.save_in_dict(saved_model, optimizer, epoch, network_type,
                               name_append="_unscaled_" + str(ensemble_num))

        if not_changed_counter > 20:
            break

        sheduler.step()

        print('Epoch: {:02d}, Loss: {:.4f}, Validation: {:.4f}, Validation_Acc: {:.4f}'.format(
            epoch, loss, validation, val_acc))


    return saved_model, best_epoch, best_epoch_acc


"""
The main function reads in commands from the consoles and executes the script accordingly

"""

if __name__ == '__main__':

    cmdline_parser = argparse.ArgumentParser('cassette final prediction')




    cmdline_parser.add_argument('-n', '--network_type',
                                default="gru2_bohb",
                                help='network_type',
                                type=str)
    cmdline_parser.add_argument('-m', '--calibration_method',
                                default="None",
                                help='calibration_method',
                                type=str)
    cmdline_parser.add_argument('-p', '--path_to_save_models',
                                default="./models/",
                                help='path_to_save_models',
                                type=str)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=150,
                                help='number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=32,
                                help='batch_size',
                                type=int)
    cmdline_parser.add_argument('-f','--train_file', 
                                type=str, nargs='+', 
                                action='append', 
                                help='file list for training')
    cmdline_parser.add_argument('-t','--test_file', 
                                type=str, nargs='+', 
                                action='append', 
                                help='file list for testing')
    cmdline_parser.add_argument('-v', '--val_file',
                                type=str, nargs='+', 
                                action='append', 
                                help='file list for validation')
                                

    args, unknowns = cmdline_parser.parse_known_args()                                
                                

    epochs = args.epochs
    BATCH_SIZE = args.batch_size
    test_results = []
    models_for_ens = 3

    
    files_train = args.train_file_name
    files_test = args.test_file_name
    files_validate = args.validate_file_name
        

    classes = len(files_train)
    class_names = [f[0].split(".fa")[0] for f in files_train]


    calibration_method = args.calibration_method
    network_type = args.network_type
    dataset_train = database.MyOwnDatasetClass(files_train).return_data()
    dataset_test = database.MyOwnDatasetClass(files_test).return_data()
    dataset_validation = database.MyOwnDatasetClass(files_validate).return_data()
    weights = utils.calculate_weight_vector(dataset_train)
    
    

    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=dataset_validation, batch_size=BATCH_SIZE, shuffle=True)

    saver = load_save.Saver(args.path_to_save_models)

    test_list = []
    final_model = []

    for ensemble_num in range(models_for_ens):

        saved_model, best_epoch, best_epoch2 = run_net(train_loader, val_loader, weights, saver, network_type, epochs,
                                                       calibration_method, classes, ensemble_num=ensemble_num)

        test_loss, test_acc_a, test_acc_b, _ = train.test(test_loader, [saved_model], calibration_method,
                                                          classes)
        final_model.append(saved_model)

        test_results.append(test_acc_a)
        test_results.append(best_epoch)

        print("Choose model by validation loss")
        print('best Epoch: {:02d}, Test-Acc: {:.4f}'.format(best_epoch, test_acc_a))

        i = 0
        test_string = ""

        for en, val in enumerate(test_acc_b):
            test_string += ' class ' + class_names[en] + '-acc: ' + str(val) + ","
        print(test_string)

    print("##############################################################################")
    print('############################ ensemble loss ###################################')
    print("##############################################################################")

    test_loss, test_acc_a, test_acc_b, _ = train.test(test_loader, final_model, calibration_method, classes)
    test_string = ""
    print("ensemble accuracy: {:.4f}".format(test_acc_a))
    for en, val in enumerate(test_acc_b):
        test_string += ' class ' + class_names[en] + '-acc: ' + str(val) + ","
    print(test_string)
