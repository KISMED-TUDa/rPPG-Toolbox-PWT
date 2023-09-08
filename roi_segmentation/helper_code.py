import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import List, Collection, NamedTuple
# from tsmoothie.smoother import *


def segment_roi(img: np.ndarray, mesh_points: List[np.ndarray], use_convex_hull: bool = False) -> np.ndarray:
    """
    Segments a Region of Interest (ROI) out of an input image according to provided set of mesh_points.
    Returns an image where all pixels outside the ROI are blackened, isolating the ROI within the output image.
    Optionally, you can choose to use the convex hull of the mesh points to define the ROI.

    :param img: (np.ndarray): An image containing the ROI represented as a numpy ndarray.
    :param mesh_points: (List[np.ndarray]): List containing ndarrays of 2d coordinates representing the mesh points of the ROI
    :param use_convex_hull:(bool, optional, default=False): A flag that determines whether to use the convex hull of the mesh points to define the ROI.
    If set to True, the convex hull will be used; otherwise, the function will use the mesh points directly.
    :return: np.ndarray: An image represented as a numpy ndarray with all pixels blackened except the ROI
    """
    img_h, img_w = img.shape[:2]
    mask_roi = np.zeros((img_h, img_w), dtype=np.uint8)
    frame_roi = img.copy()

    for point in mesh_points:
        cv2.fillConvexPoly(mask_roi, point, (255, 255, 255, cv2.LINE_AA))

    if use_convex_hull:
        # extracted and smoothed ROI triangles with cv2.convexHulls
        contours, hierarchy = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # RETR_EXTERNAL    RETR_TREE

        # create hull array for convex hull points
        hull = []
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], clockwise=False))
        mask_roi = mask_roi.copy()
        mask_roi = cv2.drawContours(mask_roi, hull, -1, (255, 255, 255), thickness=cv2.FILLED)

    output_roi = cv2.copyTo(frame_roi, mask_roi)

    return output_roi


def resize_roi(img: np.ndarray, mesh_points: List[np.ndarray]) -> np.ndarray:
    """
    Processes an image and returns a resized image with a dimension of 36x36 pixels.
    The centroid of the ROI is oriented at the center of the resized image.

    :param img: An image containing one ROI represented as a numpy ndarray.
    :param mesh_points: List containing ndarrays of 2d coordinates of the mesh_points of the ROI
    :return: np.ndarray: resized image with a dimension 36x36
    """

    x_min = mesh_points[0][:,0].min()
    y_min = mesh_points[0][:,1].min()
    x_max = mesh_points[0][:,0].max()
    y_max = mesh_points[0][:,1].max()

    distance_max = max(x_max-x_min, y_max-y_min)

    cX, cY = calc_centroids(img)

    # crop frame to square bounding box, centered at centroid
    cropped_img = img[int(cY - distance_max/2):int(cY + distance_max/2), int(cX - distance_max/2):int(cX + distance_max/2)]

    # ToDo: untersuche die Auswirkung von verschiedenen Interpolationen (INTER_AREA, INTER_CUBIC, INTER_LINEAR)
    resized_image = cv2.resize(cropped_img, (36, 36))

    return resized_image


def get_bounding_box_coordinates(img: np.ndarray, results: NamedTuple) -> (int, int, int, int):
    """
    Processes an image and returns the minimum and maximum x and y coordinates of the bounding box of detected face.
    The bounding box is determined from the minimum and maximum occurring x and y values of all mediapipe landmarks in the face.

    :param img: An image represented as a numpy ndarray.
    :param results: A NamedTuple object with a "multi_face_landmarks" field that contains the face landmarks on each detected face.
    :return: x_min, y_min, x_max, y_max: minimum and maximum coordinates of face bounding box as integers
    """
    for face_landmarks in results.multi_face_landmarks:
        img_h, img_w = img.shape[0:2]
        x_min = img_w
        y_min = img_h
        x_max = y_max = 0
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * img_w), int(landmark.y * img_h)
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y

        # draw bounding box
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        return x_min, y_min, x_max, y_max


def calc_centroids(img: np.ndarray) -> (int, int):
    """
    Calculate centroid of each ROIs in given image and the centroid in between the calculated centroids
    Based on: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img: mask image containing a single or multiple ROIs
    :return: int, int: 2d coordinates of centroid inside a single ROI or of the centroid in between multiple ROIs
    """
    # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centroid_list = []

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        centroid_list.append((cX, cY))

        cv2.circle(img, (cX, cY), 1, (0, 255, 0), -1)

    # calculate centroid between multiple rois of total face
    if len(centroid_list) > 1:
        x_coords = [p[0] for p in centroid_list]
        y_coords = [p[1] for p in centroid_list]
        _len = len(centroid_list)
        centroid_x = int(sum(x_coords) / _len)
        centroid_y = int(sum(y_coords) / _len)
        cv2.circle(img, (centroid_x, centroid_y), 1, (0, 255, 0), -1)

        return centroid_x, centroid_y
    else:
        return cX, cY


def moving_average(list_: List[tuple], window_size: int) -> (int, int):
    """
    Apply moving average filter with provided window_size to provided list.
    The averaging is calculated among all first respectively second elements in the tuples inside the window_size.

    :param list_: list of 2D-tuples containing elements to be averaged
    :param window_size: specifies the number of elements in the list that are considered in the calculation of the averaging
    :return: int, int: Integer values of averaged elements in list
    """
    x = np.array([i[0] for i in list_])
    y = np.array([i[1] for i in list_])

    mov_avg_x = np.convolve(x, np.ones(window_size), 'valid') / window_size
    mov_avg_y = np.convolve(y, np.ones(window_size), 'valid') / window_size

    return int(mov_avg_x[-1]), int(mov_avg_y[-1])


'''
def calc_single_centroids(img: np.ndarray) -> (int, int):
    """
    source: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    :param img:
    :return:
    """
    # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate moments of binary image
    M = cv2.moments(gray_image)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return cX, cY
'''


def generate_contour_points(flexible_list: List, landmarks: Collection, img_w: int, img_h: int) -> np.ndarray:
    """Generates Contour Points in image coordinates based on a list of mediapipe landmark IDs, and parameters.
    The list of landmarks might also contain multiple landmarks describing a single point.
    For more information see implementation.

    Args:
        flexible_list (List): A list of: mediapipe landmark IDs defining a contour OR tuples of mediapipe landmarks and a weight OR ...
        landmarks (Collection): A Collection of landmarks as provided by mediapipe. Must contain above IDs.
        img_w (int): width of the image processed by mp
        img_h (int): height of the image processed by mp

    Returns:
        np.ndarray: Array of 2d image coordinates describing a polygonal contour
    """
    contour_points = list()
    for item in flexible_list:
        if type(item) == int:
            p = landmarks[item]
            contour_points.append(np.multiply([p.x, p.y], [img_w, img_h]).astype(int))
        elif type(item) == tuple and len(item) == 3:  # type=tuple and len=3
            p1 = np.array([landmarks[item[0]].x, landmarks[item[0]].y])
            p2 = np.array([landmarks[item[1]].x, landmarks[item[1]].y])
            alpha = item[2]
            p = np.multiply(p1 * alpha + p2 * (1 - alpha), [img_w, img_h]).astype(int)
            contour_points.append(p)
        else:
            p1 = np.array([landmarks[item[0]].x, landmarks[item[0]].y])
            p2 = np.array([landmarks[item[1]].x, landmarks[item[1]].y])
            alpha = item[2]
            diff = alpha * (p2 - p1)
            p0 = p1 + item[3] * diff
            p0[p0 > 1] = 1
            p0[p0 < 0] = 0
            p = np.multiply(p0, [img_w, img_h]).astype(int)
            contour_points.append(p)

    return np.array(contour_points)


def mask_poly_contour(contour_points: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Creates a binary Mask image based on polygon contour points (edges) in image coordinates
       and the mask height and width

    Args:
        contour_points (np.ndarray): Edge points of contour
        img_h (int): image height
        img_w (int): image width

    Returns:
        np.ndarray: binary mask
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], (255, 255, 255, cv2.LINE_AA))
    return mask


def get_contour_image(rgb_frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """Computes contours of masks and plots contours on rgb image

    Args:
        rgb_frame (np.ndarray): WxH RGB image
        masks (List[np.ndarray]): N WxH binary masks

    Returns:
        np.ndarray: WxH RGB image
    """
    rgb_frame_copy = rgb_frame.copy()
    for _i, _mask in enumerate(masks):
        contours, hierarchy = cv2.findContours(image=_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if contours is not None and len(contours) > 0 and len(contours[0].shape) >= 3:
            _main_contour = contours[0][:, 0, :]
            center = np.mean(_main_contour, axis=0)
            cv2.drawContours(image=rgb_frame_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(rgb_frame_copy, str(_i + 1), center.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3);
    return rgb_frame_copy


def plot_contours(rgb_frame: np.ndarray, masks: List[np.ndarray]):
    rgb_frame_copy = get_contour_image(rgb_frame, masks)

    # see the results
    plt.figure()
    plt.imshow(rgb_frame_copy)
    plt.show()


def plot_jitter(centroid_list, roi_name):
    frame_counts = [x[0] for x in centroid_list]
    x_val = [x[1][0] for x in centroid_list]
    y_val = [x[1][1] for x in centroid_list]

    # 3D plot
    # ax = plt.figure("3D plot of centroid movement of " + roi_name).add_subplot(projection='3d')
    # plt.title("3D plot of centroid movement of " + roi_name)
    # ax.plot(frame_counts, x_val, zs=y_val, label='centroid movement in (x, y)')  # zdir='z',
    # ax.set_xlabel("Number of frames")
    # ax.set_ylabel("x coordinate (in pixels)")
    # ax.set_zlabel("y coordinate (in pixels)")

    # 2D plot
    plt.figure("2D plot of centroid movement of " + roi_name)
    plt.title("2D plot of centroid movement of " + roi_name)
    plt.plot(frame_counts, x_val, 'b', label='x-coordinate')
    plt.plot(frame_counts, y_val, 'r', label='y-coordinate')
    plt.grid(axis='both', color='0.95')
    plt.xlabel("Number of frames")
    plt.ylabel("Number of pixels")
    plt.legend()

    '''
    # operate smoothing on coordinates 
    plt.figure("smoothed 2D plot of centroid movement of " + roi_name)
    smoother_px = ConvolutionSmoother(window_len=10, window_type='ones')
    smoother_py = ConvolutionSmoother(window_len=10, window_type='ones')
    smoother_px.smooth(x_val)
    smoother_py.smooth(y_val)

    # plot smoothed positions in upper subplot
    plt.plot(frame_counts, smoother_px.data[0], linestyle="-", label='px', color='blue')
    plt.plot(frame_counts, smoother_py.data[0], linestyle="-", label='py', color='red')
    plt.plot(frame_counts, smoother_px.smooth_data[0], linestyle="dashed", linewidth=2, color='blue')
    plt.plot(frame_counts, smoother_py.smooth_data[0], linestyle="dashed", linewidth=2, color='red')

    # plot standard deviation
    # low_px, up_px = smoother_px.get_intervals('sigma_interval', n_sigma=3)  # generate intervals
    # low_py, up_py = smoother_py.get_intervals('sigma_interval', n_sigma=3)
    # plt.fill_between(frame_counts, low_px[0], up_px[0], alpha=0.3)
    # plt.fill_between(frame_counts, low_py[0], up_py[0], alpha=0.3)

    plt.xlabel("Number of frames")
    plt.ylabel("Number of pixels")
    plt.grid(axis='both', color='0.95')
    plt.legend()
    '''
    # plt.ion()
    plt.show(block=False)
    plt.pause(.001)


def plot_jitter_comparison(centroid_list, centroid_list_filtered, roi_name):
    frame_counts = [x[0] for x in centroid_list]
    x_val = [x[1][0] for x in centroid_list]
    y_val = [x[1][1] for x in centroid_list]

    frame_counts_filtered = [x[0] for x in centroid_list_filtered]
    x_val_filtered = [x[1][0] for x in centroid_list_filtered]
    y_val_filtered = [x[1][1] for x in centroid_list_filtered]

    # 2D plot
    plt.figure("2D plot of centroid movement of " + roi_name)
    plt.title("2D plot of centroid movement of " + roi_name)
    plt.plot(frame_counts, x_val, 'b', label='x-coordinate')
    plt.plot(frame_counts, y_val, 'r', label='y-coordinate')
    plt.plot(frame_counts_filtered, x_val_filtered, 'g--', label='filtered x-coordinate')
    plt.plot(frame_counts_filtered, y_val_filtered, 'g--', label='filtered y-coordinate')
    plt.grid(axis='both', color='0.95')
    plt.xlabel("Number of frames")
    plt.ylabel("Number of pixels")
    plt.legend()

    plt.show(block=False)
    plt.pause(.001)


# up to 15 overlapping masks
def stack_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Produces an image containing all masks layers in bytes.
    Max number of masks is 15 because 16 bit image is used

    Args:
        masks (List[np.ndarray]): N  WxH binary masks

    Returns:
        np.ndarray: WxH 16 bit image with N embedded masks
    """
    assert len(masks) < 16
    multimask_image = np.zeros(masks[0].shape, dtype=np.uint16)
    base = np.ones_like(masks[0])
    for i, mask in enumerate(masks):
        multimask_image = np.bitwise_or(np.left_shift(np.bitwise_and(base, mask), i), multimask_image)
    return multimask_image


def retain_mask_list(multi_mask_image: np.ndarray) -> List[np.ndarray]:
    """ Takes 16bit image and retrieves binary masks. Assumes that each bit is one mask.

    Args:
        multi_mask_image (np.ndarray): 16 bit image containing N masks

    Returns:
        List[np.ndarray]: N binary images
    """
    mask_list = []
    for i in range(16):
        bitmask = np.left_shift(np.ones(multi_mask_image.shape, dtype=np.uint16), i)
        mask = np.bitwise_and(multi_mask_image, bitmask)
        out = np.zeros(multi_mask_image.shape, dtype=np.uint8)
        out[mask > 0] = 255
        if np.sum(out) == 0:
            break
        mask_list.append(out)
    return mask_list


def export_multimaskimage(multi_mask_image, filename: str):
    if not filename.endswith('.png'):
        filename += '.png'
    cv2.imwrite(filename=filename, img=multi_mask_image)


def plot_rgbt_signals(rgbt_signals, fs, face_regions_names=None):
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))
    sig_len = rgbt_signals[0].shape[0]
    time = np.arange(sig_len) / fs
    for idx in range(len(rgbt_signals)):
        axs[0].plot(time, rgbt_signals[idx][:, 0])
    axs[0].set_ylabel('red')
    for idx in range(len(rgbt_signals)):
        axs[1].plot(time, rgbt_signals[idx][:, 1])
    axs[1].set_ylabel('green')
    for idx in range(len(rgbt_signals)):
        axs[2].plot(time, rgbt_signals[idx][:, 2])
    axs[2].set_ylabel('blue')
    if face_regions_names is not None:
        axs[2].legend(face_regions_names)
    axs[2].set_xlabel('time / s')
