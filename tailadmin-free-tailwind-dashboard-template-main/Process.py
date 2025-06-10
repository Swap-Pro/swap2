import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from roboflow import Roboflow


from roboflow import Roboflow
rf = Roboflow(api_key="MNbIhwpAq2AETXba1jPA")
project = rf.workspace("samrat-sahoo").project("license-plates-us-eu")
version = project.version(1)
dataset = version.download("yolov8")

#@title Datasets
from roboflow import Roboflow
import os # Import os for checking file existence

rf = Roboflow(api_key="MNbIhwpAq2AETXba1jPA")
project = rf.workspace("samrat-sahoo").project("license-plates-us-eu")
version = project.version(1)

# Define the expected download location
download_location = "/content/license-plates-us-eu-1"
zip_path = os.path.join(download_location, "roboflow.zip")

# Check if the directory and zip file already exist before downloading
if not os.path.exists(download_location) or not os.path.exists(zip_path):
    print("Dataset not found locally, downloading...")
    dataset = version.download("yolov8")
    print("Download complete.")
else:
    print("Dataset found locally, skipping download.")
    # If the dataset is already downloaded, you might need to explicitly load it
    # depending on subsequent code's needs. For this specific error,
    # the download failing was the issue, so just ensuring it exists is the fix.
    # If you need the 'dataset' object for later steps, you might need to
    # re-initialize it or adjust the code to load from the downloaded files.
    pass


#@title 1) Load model
from roboflow import Roboflow
# load model
rf = Roboflow(api_key="MNbIhwpAq2AETXba1jPA")
project = rf.workspace().project("car-plate-mav1h-rhegv")
model = project.version(1).model

#@title 2) Model Implementation
def process_vehicle_plate(vehicle_img_path, plate_img_path, model):
    """
    Replaces a vehicle's license plate .

    Args:
        vehicle_img_path: Path to vehicle image
        plate_img_path: Path to new license plate image
        model: segmentation model
    """

    try:
        # 1. Load images
        vehicle_img = cv2.cvtColor(cv2.imread(vehicle_img_path), cv2.COLOR_BGR2RGB)
        new_lp = cv2.cvtColor(cv2.imread(plate_img_path), cv2.COLOR_BGR2RGB)

        # 2. run inference
        result = model.predict(vehicle_img_path, confidence=40).json()

        polygon_points = []
        for pred in result['predictions']:
            if pred['class'] == 'plate':
                polygon_points.extend([[p['x'], p['y']] for p in pred['points']])

        if len(polygon_points) < 4:
            raise ValueError("Plate detection failed: <4 points found")

        polygon_points = np.array(polygon_points, dtype=np.float32)
        contour = polygon_points.reshape((-1, 1, 2))
        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx_corners) == 4:
            corners = approx_corners.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.int32)

        center = np.mean(corners, axis=0)

        def sort_corners(pts, center):
            def angle_from_center(pt):
                return np.arctan2(pt[1] - center[1], pt[0] - center[0])
            return sorted(pts, key=angle_from_center)

        sorted_pts = sort_corners(corners, center)
        sorted_pts = np.array(sorted_pts, dtype=np.float32)

        top_two = sorted(sorted_pts[:2], key=lambda p: p[0])  # left to right
        bottom_two = sorted(sorted_pts[2:], key=lambda p: p[0])  # left to right

        dst_pts = np.array([
            top_two[0],
            top_two[1],
            bottom_two[1],
            bottom_two[0]
        ], dtype=np.float32)


        h, w = new_lp.shape[:2]
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_lp = cv2.warpPerspective(new_lp, M, (vehicle_img.shape[1], vehicle_img.shape[0]))

        gray = cv2.cvtColor(warped_lp, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        lp_mask = np.zeros((vehicle_img.shape[0], vehicle_img.shape[1]), dtype=np.uint8)
        polygon_contour = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(lp_mask, [polygon_contour], color=255)
        lp_mask = cv2.GaussianBlur(lp_mask, (5, 5), 0)
        _, lp_mask = cv2.threshold(lp_mask, 127, 255, cv2.THRESH_BINARY)

        # Plate removal
        inpainted = cv2.inpaint(vehicle_img, lp_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        bg = cv2.bitwise_and(inpainted, inpainted, mask=mask_inv)
        fg = cv2.bitwise_and(warped_lp, warped_lp, mask=mask)
        naive_pasted = cv2.add(bg, fg)
        # naive_pasted = cv2.cvtColor(naive_pasted, cv2.COLOR_BGR2RGB)

        # Show result
        plt.figure(figsize=(10, 6))
        plt.imshow(naive_pasted)
        plt.title(f"Vehicle with New Plate:")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error processing {vehicle_img_path}: {str(e)}")
        return None, False


# @title Test Results
# Define your paths
vehicle_img_path = "C:\\Users\\aliaa\\Downloads\\tryy\\car.jpg"
plate_img_path = "C:\\Users\\aliaa\\Downloads\\tryy\\Italy.jpg"
output_path = "C:\\Users\\aliaa\\Downloads\\tryy\\output.jpg"
model = model

def process_vehicle_plate_api(vehicle_img_bytes, plate_img_path, output_dir):
    try:
        import tempfile
        import uuid
        temp_vehicle_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
        with open(temp_vehicle_path, 'wb') as f:
            f.write(vehicle_img_bytes)

        # Call the core function
        vehicle_img = cv2.cvtColor(cv2.imread(temp_vehicle_path), cv2.COLOR_BGR2RGB)
        new_lp = cv2.cvtColor(cv2.imread(plate_img_path), cv2.COLOR_BGR2RGB)

        result = model.predict(temp_vehicle_path, confidence=40).json()
        # (Add exact logic from your function `process_vehicle_plate()` here...)

        # Save output
        output_path = os.path.join(output_dir, f"result_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(naive_pasted, cv2.COLOR_RGB2BGR))
        return output_path
    except Exception as e:
        print("API processing error:", e)
        return None

process_vehicle_plate(vehicle_img_path, plate_img_path, model)