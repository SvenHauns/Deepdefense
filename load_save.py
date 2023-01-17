import torch

class Saver():


    def __init__(self, model_path):
    
        self.model_path = model_path

    def save_in_dict(self,saved_model, optimizer, epoch, network_type, temperature = None, name_append = None):
    
    
        path_save_model = self.model_path

        append_string = name_append
        if name_append == None:
            append_string = "_"

        if name_append == None:
            name_append = ""


        adjust_name = "_"
        model = path_save_model + "model" + "_" + network_type + append_string +  adjust_name + ".pt"
        
        if optimizer != None: optimizer = optimizer.state_dict()


        torch.save({'model_type':network_type,
                    'epoch' : epoch,
                    'model': saved_model.state_dict(),
                    'optimizer': optimizer,
                    'temperature':temperature},
                    model)

        return


