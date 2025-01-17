import numpy as np
from scipy.spatial.distance import cdist
import time
import cv2

def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def mean_shift_manual(data, bandwidth=2.0, max_iter=1, tol=1e-3):
    start_time = time.time()
    data = np.array(data)
    n_points, n_features = data.shape
    shifted_points = data.copy()

    for iter_num in range(max_iter):
        new_points = np.zeros_like(shifted_points)
        for i, point in enumerate(shifted_points):
            distances = np.linalg.norm(shifted_points - point, axis=1)
            weights = gaussian_kernel(distances, bandwidth)
            new_points[i] = np.sum(weights[:, None] * shifted_points, axis=0) / np.sum(weights)

        max_shift = np.linalg.norm(new_points - shifted_points, axis=1).max()
        shifted_points = new_points

        if max_shift < tol:
            print(f"Converged in {iter_num + 1} iterations.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Mean Shift completed in {total_time:.2f} seconds.")

    # Cluster centers and labels
    unique_points, labels = np.unique(shifted_points, axis=0, return_inverse=True)
    return unique_points, labels

# Example usage with an image
def mean_shift_image(path, bandwidth=2.0, max_iter=1, tol=1e-3):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_image = img.reshape((-1, 3))

    cluster_centers, labels = mean_shift_manual(flat_image, bandwidth, max_iter, tol)
    segmented_image = cluster_centers[labels].reshape(img.shape).astype(np.uint8)

    return segmented_image

# Example usage
if __name__ == "__main__":
    path = "2283107.jpeg"  # Replace with your image path
    bandwidth = 30.0
    segmented_image = mean_shift_image(path, bandwidth=bandwidth)

    # Display segmented image
    import matplotlib.pyplot as plt
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.title("Segmented Image")
    plt.show()