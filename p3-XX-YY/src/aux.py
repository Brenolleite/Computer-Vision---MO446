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



A = np.array(A)
b = np.array(b)
At = np.transpose(A)

# Verify inverse of matrix
inverted = None
try:
    inverted = inv(np.dot(At, A))

    d = np.dot(At,b)
    d = np.dot(inverted, d)

    # Adding u,v to kp
    flows.append((d[0,0], d[1,0]))
except np.linalg.linalg.LinAlgError as err:
    # Set kp to -1 (later it will be removed)
    kp[i,:] = [-1, -1]

    # Adding u,v to future ignored point (0,0)
    flows.append((0, 0))
