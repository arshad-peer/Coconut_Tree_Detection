# Coconut_Tree_Detection

**Problem Statement**:
1.	Find the no. of rows and columns of trees in the image. To justify this, you should be able to detect the lines of the rows and columns. Hence, you need to produce an intermediate output wherein you have only lines detecting those rows and columns. Also, code to count them.
2.	Find the total no. of trees in the image. Here also, you need to produce an intermediate output wherein you just have the centres of the trees. Also, a code to count them
---
**Step-by-Step Solution**:

**Step 1:**  
The number of pixels (area) per tree is visually analyzed using astronomical data visualization software (SAOImageDS9). This helps set the `min_area` and `max_area` parameters for the contour operation.

**Step 2:**  
The input image is read in two formats:  
- **Grayscale** for visualization and display.  
- **RGB** for vegetation index calculation.

**Step 3:**  
The **Vegetation Difference Vegetation Index (VDVI)** is generated from the RGB image:  
> _Note: If NIR data were available, NDVI would be used instead._  
A threshold (`V_thresh_factor`) is applied to create a binary **vegetation mask**.

**Step 4:**  
The RGB image is converted to the **YCrCb** color space.  
- Only the **Y (luminance)** channel is processed.  
- **Histogram Equalization** is applied to enhance contrast for better object separation [1].

**Step 5:**  
The **equalized Y channel** is multiplied by the **vegetation mask**.  
- This suppresses non-vegetated regions.  
- Enhances tree canopy features.

**Step 6:**  
A **Gaussian blur** is applied to the combined image to smoothen noise and small artifacts before binary conversion.

**Step 7:**  
The blurred image is **thresholded** to produce a **binary mask** for isolating individual tree canopies.

**Step 8:**  
**Contours** are extracted from the binary image.  
- **Bounding boxes** and **centroids** are calculated for each valid region.  
- Only regions within the `min_area` and `max_area` thresholds are considered coconut trees.

**Step 9:**  
Extracted `centroid_x` and `centroid_y` values are **clustered** using a `tolerance` factor.  
- These clusters help estimate **row** (horizontal) and **column** (vertical) lines,  
- Visually representing the plantation layout.

**Step 10 (Alternative Approach):**  
**Morphological operations** like `MORPH_CLOSE` or `MORPH_OPEN` are applied to:  
- Fill small gaps,  
- Connect broken canopies.  
After this, the **same contour detection process** is applied again.  

> **Note:** This approach improves detection for trees near the **edges or boundaries** of the image.

# 📤 Output

### ⭐ Without Morphing

- **Detected trees**: `37`  
- **No. of detected rows**: `8`  
- **No. of detected columns**: `7`


### ✅ With Morphological Operation (`MORPH_CLOSE`)

- **Detected trees**: `35`  
- **No. of detected rows**: `8`  
- **No. of detected columns**: `8`

# 📖 Reference

[1]	S. H. Al Mansoori, A. Kunhu, and H. Al Ahmad, 
“**Automatic palm trees detection from multispectral UAV data using normalized difference vegetation index and circular Hough transform**,” in High-Performance Computing in Geoscience and Remote Sensing VIII, B. Huang, S. López, and Z. Wu, Eds., Berlin, Germany: SPIE, Oct. 2018, p. 3. doi: 10.1117/12.2325732.


# 📚 Libraries Used

OpenCV

Numpy

Pandas

Matplotlib

# 🚀 How to Run the Algorithm

### 1. Install Dependencies

Ensure you have Python 3.8 or above installed on your system. Then install the required Python packages using the following command:

```bash
pip install -r requirements.txt 
```
### 2. Prepare the Input Image

Place your RGB input image inside the `data/` folder given

### 3. Run the Algorithm

Use the following command to execute the script:

```bash
python coconut_tree_detection.py
```
### 4. Output

After successful execution:

- The centroid coordinates of detected coconut trees will be saved as:

    `output/coconut_tree_centroids.csv`