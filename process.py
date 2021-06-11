import cv2


image = cv2.imread(input('Enter the full name/path of the car image: '))
# Converting to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
canny_edge = cv2.Canny(gray_image, 120, 200)

# Finding contours
contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]

#for debugging contours
# imgcpy = image.copy()
# cv2.drawContours(imgcpy, contours, -1, (0,255,0), 1)
# cv2.imshow('contours', imgcpy)

contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

# Extraction of the plate only
for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4: #see whether it is a Rect
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            if w/h < 1: continue #as the aspect ratio of the number plate will be wider
            license_plate = gray_image[y + 2:y + h - 2, x + 2:x + w - 2]
            break

# Noise removal
license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
(thresh, license_plate) = cv2.threshold(license_plate, 120, 180, cv2.THRESH_BINARY)

shape = license_plate.shape #dimensions of the license plate

# Display
cv2.imshow("License Plate Detection", cv2.resize(license_plate, (100*shape[1]//shape[0], 100)))
cv2.waitKey(0)

shape = (50*shape[1]//shape[0], 50)
license_plate = cv2.resize(license_plate, shape)
shape = shape[::-1] #swapping
# print("shapes equal? ", shape == license_plate.shape, shape)
