import rank as r
import numpy as np
import copy as cp

def test_weights(DS, weigths):
    total = 0
    for t in range(0, len(DS), 5):
        # Get initial index of class
        class_s = int((t+1)/5)
        total_class = 0

        for i in range(class_s, class_s + 2):
            if i != t:
                # Get distance for the class
                total_class += r.compare_regions(DS[t][1], DS[i][1], weigths[0:3].append(1), weigths[3:])

        # Sum distance for all classes
        total += total_class/2

    return total/8

def find_best_weigths():
    r.load_features_DS(True)

    DS = r.DS

    # Comparing image query with datasets
    attempts = []
    # 3 for dist_w and 5 for feat_w
    weigths = [-1, 0, 0, 0, 0, 0, 0, 0]

    ix = n = total = 0
    weigths_m = []
    mi = 9999999999
    while n < 3**len(weigths):
        weigths[ix] += 1
        if weigths[ix] > 2:
            weigths[ix] = 0
            ix -= 1
        else:
            n += 1

            if ix < len(weigths) - 1:
                ix += 1

            if(np.sum(weigths[3:]) != 0):
                print("{0} - Testing {1}".format(n, weigths), end='')
                result = test_weights(DS, weigths)
                print(" got {0} ".format(result), end='')
                attempts.append([weigths, result])

                if result < mi:
                    mi = result
                    weigths_m = weigths

                print("Min -> {0}  {1}".format(mi, weigths_m))