from sklearn import metrics
# form some_lib import cool_metric

class ClassificationMetrics:
    """
    Easy to extand class with metrics
    """
    def __init__(self) -> None:
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "recall": self._recall,
            "precision": self._precision,
            "logloss": self._logloss,
            "auc": self._auc
            #"cool_metric": self._cool_metric
        }
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception('Metric not implemented')
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true, y_proba)
            else:
                raise Exception('y_proba cannot be None for AUC')
        return self.metrics[metric](y_true, y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return metrics.f1_score(y_true, y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return metrics.recall_score(y_true, y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return metrics.precision_score(y_true, y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return metrics.log_loss(y_true, y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return metrics.roc_auc_score(y_true, y_pred)

    @staticmethod
    def _cool_metric(y_true, y_pred):
        pass
        # return _cool_metric(y_true, y_pred)

    @staticmethod
    def _my_own_metric(y_true, y_pred):
        pass # quantum tech AI metric


if __name__ =="__main__":
    y_true = [ 0, 0, 1, 0, 1, 1, 0]
    y_pred = [ 0, 0, 1, 0, 1, 1, 1]
    y_prob = [.1,.1,.5,.2,.6,.7,.8]
    # y_prob = [.5,.5,.5,.5,.5,.5,.5 ]
    cm  = ClassificationMetrics()
    for metric in cm.metrics.keys():
        print(metric, cm(metric, y_true,y_pred, y_proba=y_prob))