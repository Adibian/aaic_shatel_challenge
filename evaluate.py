from data import ShutelDataset
from model import restore_model
from utils import calculate_metrices
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# random seed setting
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def test(config, args):
    print("Prepare data ...")
    if args.test_or_val == "test":
        test_dataset = ShutelDataset(config["test"]["processed_test_data_path"], True)
    else:
        test_dataset = ShutelDataset(config["train"]["processed_val_data_path"], False)
    print(f"Number of test data: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config["test"]["batch_size"])

    print("Create model ...")
    model = restore_model(config)

    print("Predict labels for test data")
    all_predicted, all_labels = [], []
    for batch in tqdm(test_dataloader):
        input_tensor = batch["inputs"].to(config["device"])
        if "labels" in batch:
            all_labels.extend([batch["labels"][i].item() for i in range(len(batch["inputs"]))])
        else:
            all_labels.extend(["?" for i in range(len(batch["inputs"]))])

        output = model(input_tensor)  
        prdicted_labels = output > args.threshould
        all_predicted.extend(prdicted_labels.int().tolist())
    
    if args.test_or_val == "val":
        all_predicted = torch.tensor(all_predicted)
        all_labels = torch.tensor(all_labels)
        accuracy, precision, recall, f1_score = calculate_metrices(all_predicted, all_labels)
        metrices = {"accuracy":accuracy, "precision":precision, "recall":recall, "f1_score":f1_score}
        print("")
        print("#"*100)
        print("Val | " + " | ".join(f"{k}: {round(v, 3)}" for k,v in metrices.items()))
        print("#"*100)
    
    print("Save predicted labels in " + config["test"]["result_path"])
    with open(config["test"]["result_path"], "w") as f:
        f.writelines([str(row) + "\n" for row in all_predicted])

if __name__ == "__main__":
    import yaml, argparse

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_or_val", default="test", type=str)
    parser.add_argument("--threshould", default=0.5, type=float)
    args = parser.parse_args()

    test(config, args)

"""
python evaluate.py --test_or_val "val" --threshould 0.5
"""