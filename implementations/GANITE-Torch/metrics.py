import numpy as np
from sklearn.metrics import roc_auc_score

def PEHE(y, y_hat):
  """Compute Precision in Estimation of Heterogeneous Effect.

  Args:
    - y: potential outcomes
    - y_hat: estimated potential outcomes

  Returns:
    - PEHE_val: computed PEHE
  """
  PEHE_val = np.mean( np.abs( (y[:,1] - y[:,0]) - (y_hat[:,1] - y_hat[:,0]) ))
  return PEHE_val


def ATE(y,y_hat):
    """Compute Average Treatment Effect.

    Args:
      - y: potential outcomes
      - y_hat: estimated potential outcomes

    Returns:
      - ATE_val: computed ATE
    """
    ATE_val = np.abs(np.mean(y[:,1] - y[:,0]) - np.mean(y_hat[:,1] - y_hat[:,0]))
    return ATE_val

def AUC(y, y_hat):
    if np.unique(y).shape[0]<=1:
        return  0
    return  roc_auc_score(y_true=y, y_score=y_hat[:, 0])