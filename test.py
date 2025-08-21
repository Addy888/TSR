import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model # type: ignore
#############################################
 
frameWidth = 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
try:
    model = load_model("model.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
 
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize to 0-1 range
    return img

def getClassName(classNo):  # Fixed function name typo
    class_names = {
        0: 'Speed Limit 20 km/h',
        1: 'Speed Limit 30 km/h',
        2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h',
        4: 'Speed Limit 70 km/h',
        5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h',
        7: 'Speed Limit 100 km/h',
        8: 'Speed Limit 120 km/h',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',  # Fixed typo
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',  # Fixed typo
        16: 'Vehicles over 3.5 metric tons prohibited',  # Fixed typo
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'  # Fixed typo
    }
    return class_names.get(classNo, 'Unknown')

# Check if camera is opened
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

try:
    while True:
        # READ IMAGE
        success, imgOriginal = cap.read()  # Fixed variable name typo
        
        if not success:
            print("Error: Could not read frame from camera")
            break
        
        # PROCESS IMAGE
        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        
        # Show processed image (converted back to display format)
        processed_display = (img * 255).astype(np.uint8)
        cv2.imshow("Processed Image", processed_display)
        
        img = img.reshape(1, 32, 32, 1)
        
        # PREDICT IMAGE
        predictions = model.predict(img, verbose=0)  # Removed duplicate prediction line
        classIndex = np.argmax(predictions, axis=1)[0]
        probabilityValue = np.amax(predictions)
        
        # Only display result if probability is above threshold
        if probabilityValue > threshold:
            class_name = getClassName(classIndex)  # Fixed function name
            cv2.putText(imgOriginal, "CLASS: " + class_name, (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(imgOriginal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(imgOriginal, "CLASS: No sign detected", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOriginal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Result", imgOriginal)

        # Break loop on 'q' key press
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    cv2.destroyAllWindows()
    cap.release()
    print("Camera released and windows closed")