from data import ShutelDataset
from model import create_model
from utils import one_batch_processe, logging_metrices

from torch.utils.data import DataLoader, random_split
import torch, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# random seed setting
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def train(config):
    print("Prepare data ...")
    train_dataset = ShutelDataset(config["train"]["processed_train_data_path"], False)
    val_dataset = ShutelDataset(config["train"]["processed_val_data_path"], False)
    print(f"Number of train data: {len(train_dataset)}")
    print(f"Number of val data: {len(val_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"])
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"])
    
    print("Create model ...")
    model, optimizer, scheduler = create_model(config)

    loss_fn = torch.nn.BCELoss()
    
    logger = SummaryWriter(config["train"]["log_path"])
    os.makedirs(config["train"]["ckpt_path"], exist_ok=True)

    total_step = config["train"]["total_step"]
    val_step = config["train"]["val_step"]
    save_step = config["train"]["save_step"]
    epoch, step = 1, 1

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = step-1
    outer_bar.update()
    train_log_values = {"loss":0, "accuracy":0, "precision":0, "recall":0, "f1_score":0}
    val_log_values = {"loss":0, "accuracy":0, "precision":0, "recall":0, "f1_score":0}

    ## Training
    while True:
        inner_bar = tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch), position=1)
        for batch in train_dataloader:
            train_log_values = one_batch_processe(batch, model, loss_fn, optimizer, config, train_log_values, train=True)

            if step % val_step == 0:
                print("Evaluate model ...")
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_log_values = one_batch_processe(batch, model, loss_fn, None, config, val_log_values, train=False)
                train_log_values = {k:v/val_step for k,v in train_log_values.items()}
                val_log_values = {k:v/len(val_dataloader) for k,v in val_log_values.items()}
                
                logging_metrices(train_log_values, val_log_values, logger, step)
                if scheduler:
                    scheduler.step(train_log_values["loss"])

                train_log_values = {"loss":0, "accuracy":0, "precision":0, "recall":0, "f1_score":0}
                val_log_values = {"loss":0, "accuracy":0, "precision":0, "recall":0, "f1_score":0}

                model.train()
            
            if step % save_step == 0:
                print("Save model checkpoint...")
                torch.save(
                    {"model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), \
                     "optimizer": optimizer.state_dict()},
                    os.path.join(config["train"]["ckpt_path"], "{}.pth.tar".format(step))
                )
            if step == total_step:
                quit()

            step += 1
            outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1

if __name__ == "__main__":
    import yaml
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    train(config)