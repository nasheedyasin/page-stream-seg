import json
from typing import List

from helper_classes import Document


class SegEval:
    """This class wraps all the segmentation (document level)
    evaluation metrics.
    """
    def __init__(self, gt_docs: List[Document],
                 pred_docs: List[Document]) -> None:
        """
        Args:
            gt_docs (List[Document]): The ground truth documents.
            pred_docs (List[Document]): The predicted documents.
        """
        # I'm converting to `set` to prevent any duplicate entries in
        # the data.
        self.__gt_docs = set(gt_docs)
        self.__pred_docs = set(pred_docs)

    @property
    def gt_docs(self):
        return self.__gt_docs

    @property
    def pred_docs(self):
        return self.__pred_docs

    def __strict_iou(self) -> float:
        """Gives the strict IoU (between 1 and 0) score between the
        true and predicted documents. This metric treats any deviations
        from the ground truth described document bounds as an erroneous
        prediction.
        

        Returns:
            float: Strict IoU score (`S-IoU`).
        """
        return len(self.__gt_docs.intersection(self.__pred_docs)) / len(
            self.__gt_docs.union(self.__pred_docs))

    def __strict_precision(self) -> float:
        """Gives the strict precision (between 1 and 0) score between the
        true and predicted documents. This metric treats any deviations
        from the ground truth described document bounds as an erroneous
        prediction.
        

        Returns:
            float: Strict precision (`S-Precision`).
        """
        return len(self.__gt_docs.intersection(self.__pred_docs)) / len(
            self.__pred_docs)

    def __strict_recall(self) -> float:
        """Gives the strict recall (between 1 and 0) score between the
        true and predicted documents. This metric treats any deviations
        from the ground truth described document bounds as an erroneous
        prediction.
        

        Returns:
            float: Strict recall (`S-Recall`).
        """
        return len(self.__gt_docs.intersection(self.__pred_docs)) / len(
            self.__gt_docs)

    def __strict_f1(self) -> float:
        """Gives the strict f1 (between 1 and 0) score between the
        true and predicted documents. This metric treats any deviations
        from the ground truth described document bounds as an erroneous
        prediction.
        

        Returns:
            float: Strict recall (`S-F1`).
        """
        s_precision = self.__strict_precision()
        s_recall = self.__strict_recall()

        return 2 * ((s_precision * s_recall) / (s_precision + s_recall))

    def __call__(self) -> str:
        """
        Returns:
            Dict[str, float]: Returns a `Json`-ised string version of all
            the segmentation metrics.
        """
        return json.dumps({
            'strictF1': self.__strict_f1(),
            'strictIou': self.__strict_iou(),
            'strictRecall': self.__strict_recall(),
            'strictPrecision': self.__strict_precision()
        })
