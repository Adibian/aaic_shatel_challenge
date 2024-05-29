import torch
import torch.nn as nn
import os


class ShutelModel(nn.Module):
    def __init__(self, config):
        super(ShutelModel, self).__init__()
        hidden_dim = config["model"]["hidden_dim"]
        self.lstm = nn.LSTM(6, hidden_dim, num_layers=config["model"]["lstm_layers"], bidirectional=False, batch_first=True, dropout=config["model"]["dropout"])
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//4, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, input_tensor):
        rnn_out, _ = self.lstm(input_tensor)
        rnn_out = rnn_out[:,-1,:]

        outputs = self.linear(rnn_out)
        outputs = self.activation(outputs)
        return outputs.squeeze(-1)
    
def create_model(config):
    model = ShutelModel(config).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["optimizer_lr"])
    return model, optimizer, None

def restore_model(config):
    model = ShutelModel(config).to(config["device"])
    ckpt_path = os.path.join(config["train"]["ckpt_path"], str(config["test"]["restore_step"])+".pth.tar")
    ckpt = torch.load(ckpt_path, map_location=torch.device(config["device"]))
    model.load_state_dict(ckpt["model"])
    return model
