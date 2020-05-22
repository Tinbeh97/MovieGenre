from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
from sklearn import metrics

def prauc(y_true, y_pred, end, thresh = 0):
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = y_true.shape[-1]
    y_predict = np.zeros((y_pred.shape[0],n_classes))
    for i in range(n_classes):
        
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        if(end):
            optimal_threshold = thresh
        else:
            fpr, tpr, thresholds = metrics.roc_curve(y_true[:, i],y_pred[:, i])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        y_predict[:,i] = (y_pred[:, i] > optimal_threshold).astype(int)
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
                                                                    y_pred.ravel())

    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                         average="micro")
    average_precision["macro"] = average_precision_score(y_true, y_pred,
                                                         average="macro")
    average_precision["weighted"] = average_precision_score(y_true, y_pred,
                                                         average="weighted")
    return average_precision, y_predict, optimal_threshold
            