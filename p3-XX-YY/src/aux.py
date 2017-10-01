# Transform the keypoints into the new coordinates using the (u, v) solutions
# and interpolate the keypoints into valid coordinates
def interpolate(kp, solution):

    # Run over all the keypoints
    for i in range(len(kp)):
        x = kp[i][0]
        y = kp[i][1]

        u = solution[i][0]
        v = solution[i][1]

        x = math.floor(x + u)
        y = math.floor(y + u)

        kp[i] = (x, y)

    return kp
