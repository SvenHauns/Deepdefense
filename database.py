import utils

"""
The class MyOwnDatasetClass 

arguments:
files: files to be read in 

functions:
process: main logic of file processing
return_data: returns processed data list

"""


class MyOwnDatasetClass:
    def __init__(self, files):

        self.files = files
        self.data = self.process()

    def return_data(self):

        return self.data

    def process(self):
        data_list = []
        protein_type = 0

        for file_ in self.files:

            data_str = utils.load_protein_data(file_)

            for prot_string in data_str:
                data_list.append([prot_string, protein_type])
            protein_type = protein_type + 1

        return data_list
