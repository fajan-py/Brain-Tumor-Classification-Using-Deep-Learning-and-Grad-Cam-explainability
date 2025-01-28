from sklearn.metrics import accuracy_score, f1_score
import torch


def evaluate_model(model, dataloader, device):

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            data, targets = batch
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_accuracy = accuracy_score(all_targets, all_preds)
    avg_f1_score = f1_score(all_targets, all_preds, average='weighted')

    return avg_accuracy, avg_f1_score
