import rank as r
import numpy as np

def find_best_weigths():
    r.load_features_DS(True)

    DS = r.DS

    # Comparing image query with datasets
    Attempts = []
    # 3 for dist_w and 5 for feat_w
    dist_w = [1, 1, 1, 1]
    feat_w = [1, 1, 1, 1, 1]

    total = 0
    for x in range(3**len(dist_w)-1):

        for t in range(len(DS)):
            # Get initial index of class
            class_s = int((t+1)/5)
            total_class = 0

            for i in range(class_s, class_s + 5, 2):
                if i != t:
                    # Get distance for the class
                    total_class += r.compare_regions(DS[t][1], DS[i][1], dist_w, feat_w)
                    #total_class += r.compare_regions(DS[t][1], DS[i][1], weigths[0:4], weigths[4:])

            # Sum distance for all classes
            total += total_class/4

        Attempts.append([dist_w, feat_w, total/40])

    return sorted(sorted(Attempts, key=lambda x: x[2]))

print(find_best_weigths())