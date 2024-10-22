from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def iou_metric(preds, labels, num_classes=2):
    """ Calculate iou

    Args:
        preds : the predictions of the model
        labels : the actual labels

    Returns:
        float: calculated IoU    
    """
    # preds = preds.squeeze(0)
    
    iou_per_class = []
    
    for cls in range(num_classes):
        true_cls = (labels == cls)
        pred_cls = (preds == cls)
        intersection = (true_cls & pred_cls).sum().item()
        union = (true_cls | pred_cls).sum().item()
        iou = intersection / union if union != 0 else 0
        iou_per_class.append(iou)
    
    total_intersection = (preds & labels).sum().item()
    total_union = (preds | labels).sum().item()
    
    global_iou = total_intersection / total_union if total_union != 0 else 0
    return global_iou, iou_per_class
    
def f1_metric(preds, labels):
    """ Calculate f1 metric

    Args:
        preds : the predictions of the model
        labels : the actual labels

    Returns:
        float: calculated f1 score    
    """
    # Flatten the arrays
    y_pred_flat = preds.cpu().numpy().flatten()
    y_true_flat = labels.cpu().numpy().flatten()
    
    # Calculate F1 score
    f1 = f1_score(y_pred_flat, y_true_flat, zero_division=0)
    return f1

def precision_metric(preds, labels):
    """ Calculate precision

    Args:
        preds : the predictions of the model
        labels : the actual labels

    Returns:
        float: calculated Precision    
    """
    y_pred_flat = preds.cpu().numpy().flatten()
    y_true_flat = labels.cpu().numpy().flatten()
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    return precision

def recall_metric(preds, labels):
    """ Calculate Recall

    Args:
        preds : the predictions of the model
        labels : the actual labels

    Returns:
        float: calculated Recall    
    """
    y_pred_flat = preds.cpu().numpy().flatten()
    y_true_flat = labels.cpu().numpy().flatten()
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    return recall

def accuracy_metric(preds, labels, num_classes=2):
    """Calculate accuracy

    Args:
        preds : the predictions of the model
        labels : the actual labels

    Returns:
        float: calculated Accuracy
    """
    per_class_acc = []
    # Total pixels for accucary
    total_pixels = torch.numel(labels)
    for cls in range(num_classes):
        true_cls = (labels == cls)
        pred_cls = (preds == cls)
        correct_pixels = (true_cls & pred_cls).sum().item()
        total_class_pixels = true_cls.sum().item()
        accuracy = correct_pixels / (total_class_pixels + 1e-10)
        per_class_acc.append(accuracy)
    
    # Correct predictions for accuracy
    total_correct_pixels = (preds == labels).sum().item()
    
    global_accuracy = total_correct_pixels / (total_pixels + 1e-10)
    return global_accuracy, per_class_acc