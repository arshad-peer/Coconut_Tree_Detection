import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

def read_image(img_path):
    img_8_bit = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    return img_8_bit, img_rgb

def compute_vdvi(img_rgb, V_thresh_factor):
    r_img = img_rgb[:, :, 0].astype(float)
    g_img = img_rgb[:, :, 1].astype(float)
    b_img = img_rgb[:, :, 2].astype(float)
    vdvi = (2 * g_img - r_img - b_img) / (2 * g_img + r_img + b_img + 1e-6) # compute VDVI
    vdvi_norm = cv2.normalize(vdvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    veg_mask = (vdvi > V_thresh_factor).astype(np.uint8) # create binary vegetation mask 
    # plt.imshow(vdvi, cmap='gray')
    # plt.title("VDVI")
    # plt.show()
    # plt.imshow(veg_mask, cmap='gray')
    # plt.title("VDVI_binary_mask")
    # plt.show()
    return vdvi, vdvi_norm, veg_mask

def rgb_to_ycbcr(img_rgb):
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y_channel, Cr_channel, Cb_channel = cv2.split(img_ycbcr) # split channels
    Y_eq = cv2.equalizeHist(Y_channel) # apply histogram equalization on Y channel
    return Y_eq

def combine_y_veg_mask(Y_eq, veg_mask):
    Y_eq_norm = Y_eq / 255.0
    Y_veg_mask = (Y_eq_norm * veg_mask) * 255 
    Y_veg_mask = Y_veg_mask.astype(np.uint8)
    # # print(np.unique(Y_veg_mask))
    # plt.imshow(Y_veg_mask, cmap='gray')
    # plt.title("Combine_Y_veg_mask")
    # plt.show()
    return Y_veg_mask

def blur_y_mask(Y_veg_mask):
    Y_veg_mask_blur = cv2.GaussianBlur(Y_veg_mask, (5, 5), 1.0)
    # # print(np.unique(Y_veg_mask_blur))
    # plt.imshow(Y_veg_mask_blur, cmap='gray')
    # plt.title("Combine_Y_veg_mask_blur")
    # plt.show()
    return Y_veg_mask_blur

def create_binary(Y_veg_mask_blur, B_thresh_factor):
    thresh = cv2.threshold(Y_veg_mask_blur, B_thresh_factor, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(thresh, cmap='gray')
    # plt.title("Binary_thresholded_image")
    # plt.show()
    return thresh

def morphology_ops(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    plt.imshow(morphed, cmap='gray')
    plt.title("Morphed image")
    plt.show()
    return morphed

def detect_contour(morphed, min_area, max_area, img_8_bit):
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    centroids = []
    img_color = cv2.cvtColor(img_8_bit, cv2.COLOR_GRAY2BGR)

    feature_id = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            bounding_boxes.append((x, y, w, h))
            centroids.append({
                "tree_ID": feature_id,
                "centroid_x": cx,
                "centroid_y": cy
            })
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 10)  # red box
            # cv2.circle(img_color, (cx, cy), 10, (255, 255, 0), -1)  # yellow filled circle
            feature_id += 1

    centroids_df = pd.DataFrame(centroids)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    for c in centroids:
        plt.scatter(c["centroid_x"], c["centroid_y"], color='yellow', s=40, edgecolors='black', linewidths=1)
    plt.title("Bounding Boxes for Coconut Trees")
    plt.show()

    return bounding_boxes, centroids_df

def cluster_positions(positions, tolerance=100):
    positions = sorted(positions)
    clustered = []
    cluster = [positions[0]]

    for pos in positions[1:]:
        if abs(pos - cluster[-1]) <= tolerance:
            cluster.append(pos)
        else:
            clustered.append(int(np.mean(cluster)))
            cluster = [pos]
    clustered.append(int(np.mean(cluster)))
    return clustered

def find_row_column_lines(y_centers, x_centers, img_8_bit):
    img_color = cv2.cvtColor(img_8_bit, cv2.COLOR_GRAY2BGR)
    # cluster centers to find approximate rows and columns
    row_lines = cluster_positions(y_centers, tolerance=150)
    # print("no of detected rows:", len(row_lines))
    col_lines = cluster_positions(x_centers, tolerance=200)
    # print("no of detected columns:", len(col_lines))

    # draw horizontal lines for rows
    for y_line in row_lines:
        cv2.line(img_color, (0, y_line), (img_color.shape[1], y_line), (0, 255, 0), 10)  # green horizontal
    # draw vertical lines for columns
    for x_line in col_lines:
        cv2.line(img_color, (x_line, 0), (x_line, img_color.shape[0]), (255, 0, 0), 10)  # blue vertical

    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Number Row and Column Lines")
    plt.show()

    return row_lines, col_lines

def process_image(img_path, V_thresh_factor, B_thresh_factor, min_area, max_area):
    # read input image
    img_8_bit, img_rgb = read_image(img_path)
    # vdvi generation
    vdvi, vdvi_norm, veg_mask = compute_vdvi(img_rgb, V_thresh_factor)
    # rbg to ycbcr to seperate luminance channel
    Y_eq = rgb_to_ycbcr(img_rgb)
    # merge y channel and veg mask
    Y_veg_mask = combine_y_veg_mask(Y_eq, veg_mask)
    # apply gaussian blur on Y_veg_mask
    Y_veg_mask_blur = blur_y_mask(Y_veg_mask)
    # apply thresholding on Y_veg_mask_blur
    thresh = create_binary(Y_veg_mask_blur, B_thresh_factor)
    # apply MORPH_CLOSE or MORPH_OPEN (if required)
    morphed = morphology_ops(thresh)
    # obtain the bounding box of coconut trees
    bounding_boxes, centroids_df = detect_contour(morphed, min_area, max_area, img_8_bit)
    # bounding_boxes, centroids_df = detect_contour(thresh, min_area, max_area, img_8_bit)
    # # get centers of bounding boxes
    x_centers = centroids_df["centroid_x"].tolist()
    y_centers = centroids_df["centroid_y"].tolist()
    # count the no of rows and columns where the trees detected
    row_lines, col_lines = find_row_column_lines(y_centers, x_centers, img_8_bit)

    return bounding_boxes, centroids_df, row_lines, col_lines

# main function
def main():
    start_time = time.time()

    # input image path
    img_path = r'D:\Others\Galaxeye\test_image\test_image\test_image.jpg'

    # user parameters
    V_thresh_factor = 0.40  # vegetation threshold
    B_thresh_factor = 30   # binary image threshold
    # coconut tree (coconut tree)
    min_area = 20000
    max_area = 300000  

    bounding_boxes, centroids_df, row_lines, col_lines = process_image(img_path, V_thresh_factor, B_thresh_factor, min_area, max_area)
    centroids_df.to_csv("coconut_tree_centroids.csv", index=False)

    print("detected trees:", len(bounding_boxes))
    print("no of detected rows:", len(row_lines))
    print("no of detected columns:", len(col_lines))
   
    # calculate total computation time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Computation time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()  
