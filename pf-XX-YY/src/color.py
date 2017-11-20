import utils
import numpy as np
import cv2
import hough

from sklearn.metrics import pairwise_distances_argmin_min

hsvColor = []

def initColors():
    # Yellow
    hsvColor.append(30)

    # Blue
    hsvColor.append(120)

def filterLargerComponent(areas, selectedLabels, hough):
    # Dont filter if just background present
    if len(areas) <= 1 or (len(selectedLabels) == 0 and hough):
        return []

    # Sort in desc order
    if hough:
        # If using hough filter select only by chosen labels
        largest = sorted(areas[selectedLabels], reverse = True)[0]
    else:
        # If not using hough get second largest but background
        largest = sorted(areas, reverse = True)[1]

    # Get all indexes higher than threshold
    indexes = np.where(areas > largest / 3)[0]

    # Get all index different from background
    indexes = indexes[indexes != 0]

    if hough:
        # Get just selected labels from filter
        indexes = np.intersect1d(indexes, selectedLabels)

    return indexes

def houghLabelFilter(circles, labels):
    # Create array of selected colors
    selectedLabels = []

    # Get height and width
    height, width = labels.shape

    # Select all labels that intersects with circles
    for pos in circles:
        x, y, _ = pos

        # Get all the labels except background
        if labels[y][x] != 0:
            selectedLabels.append(labels[y][x])

    # Return unique labels selected
    return np.unique(selectedLabels)


# Create global variables to trace balls
# ID, centroid_x, centroid_y
last_frame_balls = []
ball_id = -1

# Function to add ball_id into ballsInfo vector
def append_id(balls, b_id, index, new = False):
    if new:
        # Add new counter to next ball
        b_id += 1

    # Transform to list for append
    balls[index] = list(balls[index])

    # Update new id to balls (transform to list)
    balls[index].append(b_id)

    # Transform to tuple to deal with other code
    balls[index] = tuple(balls[index])

    return balls, b_id

# Function to trace balls id by distance
def getBallsId(balls):
    # Define global variables
    global last_frame_balls, ball_id

    # Check if balls is not empty
    if len(balls) == 0:
        return balls

    # If exists balls in last frame
    if len(last_frame_balls) > 0:
        # Get current position and last position
        cur_pos  = np.array(balls)[:,5:7]
        last_pos = np.array(last_frame_balls)[:, 1:3]

        # Calculate distances
        idx, dist = pairwise_distances_argmin_min(cur_pos, last_pos)

        # Create dictionary using ball ids
        dic = list(zip(np.array(last_frame_balls)[idx,0], dist))

        # Create indexes to dictionary to sort
        dic = list(zip(np.arange(len(dic)), dic))

        # Sort array by distance
        dic = sorted(dic, key=lambda x: x[1][1])

        # Create or delete balls
        dic_size = len(dic)
        last_size = len(last_frame_balls)
        if dic_size > last_size:
            # Verify difference on size
            diff = dic_size - last_size

            # create new array
            new_array = dic[dic_size-diff:]

            # Update dic removing new balls
            dic = dic[:dic_size-diff]

            for item in new_array:
                # Get ball index
                index = item[0]

                # Append id to ballsInfo
                balls, ball_id = append_id(balls, ball_id, index, True)

                # Add balls to last_frame
                last_frame_balls.append([ball_id, balls[index][5], balls[index][6]])

        elif dic_size < last_size:
            # Get ids to keep
            ids = np.array(list(dict(dic).values()))[:,0]

            # Remove items not in ids
            last_frame_balls = list(filter(lambda x : x[0] in ids, last_frame_balls))

        # Loop over dictionay
        i = 0
        for item in dic:
            # Get index
            index = item[0]

            # Get id from distance
            b_id = item[1][0]

            # Get index in last frame by id
            ix = np.where(np.array(last_frame_balls)[:,0] == b_id)[0][0]

            # Update last frame centroids x,y
            last_frame_balls[ix][1] = cur_pos[index][0]
            last_frame_balls[ix][2] = cur_pos[index][1]

             # Append id to ballsInfo
            balls, _ = append_id(balls, b_id, index)

            i += 1
    # Init balls in first frame
    else:
        for i in range(len(balls)):
            # Append id to ballsInfo
            balls, ball_id = append_id(balls, ball_id, i, True)

            # Add balls to last_frame
            last_frame_balls.append([ball_id, balls[i][5], balls[i][6]])

    return balls

def detectByColor(frame, hough_active = False):
    output = []

    # Transform the color space into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i in range(len(hsvColor)):
        # Gets a black and white image, all the pixels in the color range will be
        # painted white, everything else will be black
        mask = cv2.inRange(hsv, (hsvColor[i] - 40, 86, 6), (hsvColor[i] + 60, 255, 255))

        # Remove some noise from image
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 2)
        mask = cv2.dilate(mask, kernel, iterations = 2)

        # Gets all the connected components in the mask
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        # Create selected labels by hough filter
        selectedLabels = []
        if hough_active:
            # Fit a circle in the mask
            circles = hough.find(mask)

            # Filter color using hough circles
            if len(circles) > 0:
                frame = hough.draw(frame, circles)
                selectedLabels = houghLabelFilter(circles, labels)

        # Gets the index of the largest connected component
        # stats[:, 4] gets all the areas of components
        indexes = filterLargerComponent(stats[:, 4], selectedLabels, hough_active)

        # Creating ball information
        for ix in indexes:
            x1, y1, x2, y2, _    = stats[ix]
            output.append((hsvColor[i], x1, y1, x1 + x2, y1 + y2, int(centroids[ix][0]), int(centroids[ix][1])))

        #  maskJoin = cv2.bitwise_or(maskJoin, mask, mask= mask)
        # wName = 'Mask' + str(i)
        # cv2.imshow(wName, mask)
        # cv2.imshow("Hough", frame)

    # Find balls IDs
    output = getBallsId(output)

    return output

initColors()
