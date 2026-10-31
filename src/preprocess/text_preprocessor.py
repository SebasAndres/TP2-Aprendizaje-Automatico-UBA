import gzip

class DatasetWrapper():

    def __init__(self, dataset_local_path:str):
        self.local_path = dataset_local_path
        self.data = gzip.open(self.local_path, mode="rt", encoding="utf-8")

    def __getitem__(self):
        return self.data.readline()