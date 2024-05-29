import torch


def calculate_metrices(prdicted_labels, labels):
    accuracy = (prdicted_labels==labels).sum() / len(labels)
    precision = torch.logical_and(prdicted_labels==1, prdicted_labels==labels).sum() / (prdicted_labels==1).sum()
    recall = torch.logical_and(labels==1, prdicted_labels==labels).sum() / (labels==1).sum()
    f1_score = 2*(precision*recall)/(precision+recall)
    return accuracy.item(), precision.item(), recall.item(), f1_score.item()

def one_batch_processe(batch, model, loss_fn, optimizer, config, log_values, train=True):
    input_tensor = batch["inputs"].to(config["device"])
    labels = batch["labels"].to(config["device"])
    
    output = model(input_tensor) 
    loss = loss_fn(output, labels)
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prdicted_labels = output>0.5
    accuracy, precision, recall, f1_score = calculate_metrices(prdicted_labels, labels)

    log_values["loss"] += loss.item()
    log_values["accuracy"] += accuracy
    log_values["precision"] += precision
    log_values["recall"] += recall
    log_values["f1_score"] += f1_score
    return log_values 


def logging_metrices(train_log_values, val_log_values, logger, step):
    for k, v in train_log_values.items():
        logger.add_scalar(f"{k}/train_{k}", v, step)
    for k, v in val_log_values.items():
        logger.add_scalar(f"{k}/val_{k}", v, step)
    
    print("")
    print("#"*100)
    print("Train | " + " | ".join(f"{k}: {round(v, 3)}" for k,v in train_log_values.items()))
    print("Val   | " + " | ".join(f"{k}: {round(v, 3)}" for k,v in val_log_values.items()))
    print("#"*100)
