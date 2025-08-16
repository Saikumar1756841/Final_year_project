import cv2
import torch

# Load YOLOv5 medium model (better accuracy)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)


# Classes in COCO dataset related to vehicles
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

def count_vehicles(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return 0

    # Convert image to RGB as YOLO expects RGB input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLOv5 model inference
    results = model(img_rgb)

    # Extract class indices from results
    labels = results.xyxyn[0][:, -1].cpu().numpy()  # class indices
    names = results.names  # class names

    # Count only vehicle classes
    count = 0
    for label in labels:
        class_name = names[int(label)]
        if class_name in vehicle_classes:
            count += 1
    return count

def main():
    directions = ['north', 'south', 'easttt', 'west']
    vehicle_counts = {}

    # Read and process each image
    for d in directions:
        img_path = f'{d}.jpg'  # Make sure these images are in the same folder
        count = count_vehicles(img_path)
        vehicle_counts[d] = count

    print("Vehicle counts per direction:", vehicle_counts)

    # Decide which lane gets green based on max vehicles
    max_dir = max(vehicle_counts, key=vehicle_counts.get)
    print(f"Give green signal to {max_dir.upper()} lane")

    # Calculate proportional green signal times (out of 60 seconds)
    total_vehicles = sum(vehicle_counts.values())
    if total_vehicles == 0:
        print("No vehicles detected on any lane. Using default timings.")
        green_time = {d: 15 for d in directions}  # 15 seconds each
    else:
        green_time = {
            d: round((vehicle_counts[d] / total_vehicles) * 60, 2)
            for d in directions
        }

    print("Proportional green times (in seconds):")
    for d in directions:
        print(f"{d.capitalize()}: {green_time[d]} seconds")

if __name__ == "__main__":
    main()
