
import numpy
from scipy import optimize
from typing import Union, List, Tuple
from helper_classes import Document as doc


# calculates iou for two documents -- allows for None-type input to account for padding
def spectrum_iou(d1: Union[doc, None], d2: Union[doc, None]) -> float:
    if d1 is None or d2 is None:
        return 0

    int_len = sum(1 for x in range(d1.start_idx, d1.end_idx + 1) if d2.start_idx <= x <= d2.end_idx + 1)

    return int_len / ((max(d1.end_idx, d2.end_idx) + 1) - min(d1.start_idx, d2.start_idx))


# takes list of "true" documents (true_list) and list of "predicted" documents (pred_list) and returns
# a tuple (L, d), where d = max(len(true_list), len(pred_list)), and L is a list of triples (x, y, iou(x, y)),
# where x is a "true" document and y is a "predicted" document -- pairs off "true"/"predicted" documents to
# maximize iou score
def hungarian_matching(true_list: List[doc], pred_list: List[doc]) -> Tuple[List[Tuple[doc, doc, float]], int]:
    # padding shorter list with None because Hungarian matching requires square matrix
    if len(true_list) > len(pred_list):
        pred_list += [None for _ in range(len(true_list) - len(pred_list))]
    elif len(pred_list) > len(true_list):
        true_list += [None for _ in range(len(pred_list) - len(true_list))]

    cost_matrix = numpy.array([[1 - spectrum_iou(d1, d2) for d1 in pred_list] for d2 in true_list])
    assign_matrix = optimize.linear_sum_assignment(cost_matrix)  # actual matching done by scipy :)
    am_0, am_1 = assign_matrix[0].tolist(), assign_matrix[1].tolist()

    # len(true_list) = max(len(true_list), len(pred_list)) because of padding
    return [(true_list[i], pred_list[j], 1 - cost_matrix[i][j]) for i, j in zip(am_0, am_1)], len(true_list)


# takes lists of "true"/"predicted" documents and returns sum(iou(x, y)) / max(len(true_list), len(pred_list))
def global_iou(true_list: List[doc], pred_list: List[doc]) -> float:
    hm_list, den = hungarian_matching(true_list, pred_list)

    return sum(x[2] for x in hm_list) / den
