import torch
from torch.utils.data import Dataset
import pickle
   

class ShutelDataset(Dataset):
    def __init__(self, processed_data_path, is_test_data=False):
        self.is_test_data = is_test_data
        with open(processed_data_path, 'rb') as handle:
            saved_data = pickle.load(handle)
            if is_test_data:
                self.data, self.contract_ids = saved_data["data"], saved_data["contract_ids"]
            else:
                self.data, self.labels, self.contract_ids = saved_data["data"], saved_data["labels"], saved_data["contract_ids"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        contract_id = self.contract_ids[idx]
        input_data = torch.from_numpy(self.data[contract_id]).float()
        if not self.is_test_data:
            label = torch.tensor(self.labels[contract_id]).float()
            return {"inputs": input_data, "labels": label}
        else:
            return {"inputs": input_data}
