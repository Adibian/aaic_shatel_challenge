import torch


def calculate_metrices(prdicted_labels, labels):
    precision_macro = precision_score(labels, prdicted_labels, average="macro", zero_division=0)
    recall_macro = recall_score(labels, prdicted_labels, average="macro", zero_division=0)
    f1_score_macro = f1_score(labels, prdicted_labels, average="macro")
    accuracy = accuracy_score(labels, prdicted_labels)
    return accuracy, precision_macro, recall_macro, f1_score_macro

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
    accuracy, precision, recall, f1_score = calculate_metrices(prdicted_labels.cpu().numpy(), real_label.cpu().numpy())

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
