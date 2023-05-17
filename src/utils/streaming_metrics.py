
# Fernando López Gavilánez, 2023

# False Alarm (FA) = False Positives (FP)
# False Rejection (FR) = False Negatives (FN) = Misses 


def measure_frame_errors(gt, predictions):
    assert len(gt) == len(predictions), "length of ground truth list and predictions must be the same"
    fp, fn = 0, 0
    for i in range(len(gt)):
        if gt[i] != predictions[i]:
            if gt[i] == 1:
                fn += 1
            else:
                fp += 1
    return fp, fn


def get_frame_error_rate(gt, predictions):
    fp, fn = measure_frame_errors(gt, predictions)
    return 100*(fp+fn)/len(gt)


def get_detect_cost_function(gt, predictions, λ=0.5):
    fp, fn = measure_frame_errors(gt, predictions)
    return (1-λ)*fn + λ*fp


def get_fer_and_dcf(gt, predictions, λ=0.5):
    fp, fn = measure_frame_errors(gt, predictions)
    fer = 100*(fp+fn)/len(gt)
    dcf = (1-λ)*fn + λ*fp
    return fer, dcf
