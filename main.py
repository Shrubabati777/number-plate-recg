from sys import stdout, exit

try:  # if numpy is not found
    import numpy as np
except Exception as e:
    print(e)
    print("Install numpy with 'pip3 install numpy' command (or pip)")
    exit(1)

try:  # if numpy is not found
    import cv2
except Exception as e:
    print(e)
    exit(1)

from recognizer import LogisticRegression as Reg

# np.set_printoptions(threshold=np.inf) #enable this while debugging

#load the trained model
WB = np.array(np.load('trained_data100.npy', allow_pickle=True))

def img_to_np(img):
    img = cv2.resize(img, (50, 50))
    img = np.array(
        img,    # np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 129,
        dtype=int)

    return img

def predict(images):
    #initialise the regressor object
    regressor = [Reg(0.0001) for _ in range(36)]

    #initialise them with the weights from the trained model
    for i in range(36):
        regressor[i].weights, regressor[i].bias = WB[i]

    result = []
    for img in images:
        img = img_to_np(img.astype('uint8'))
        # np.savetxt(stdout, img, fmt='%i')
        preds = np.array([regressor[i].predict(img.reshape(-1))
                          for i in range(36)])
        result.append('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[np.argmax(preds)])

    print('The car number is:', ''.join(result))

def main():
    #let the number plate localisation be done
    from process import shape, license_plate
    #debugging
        # np.savetxt(stdout, np.array(
        #     cv2.resize(license_plate, (20*shape[1]//shape[0],20)) > 50, 
        #     dtype=int
        #     ) , fmt='%i')

    #localise characters' images from license_plate
    # _, thresh = cv2.threshold(license_plate, 50, 155, cv2.THRESH_BINARY_INV)
    # cv2.imshow('importing lp', license_plate)
    # cv2.waitKey(0)
    license_plate = cv2.Canny(license_plate, 120, 200)
    contours, _ = cv2.findContours(license_plate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    characters = [[*cv2.boundingRect(x)] for x in contours]
    # print(characters, characters.__len__())
    # print('shape =', shape)

    # avg_w = sum([x[2] for x in characters])/len(characters)
    # avg_h = sum([x[3] for x in characters])/len(characters)

    characters = [
        *set([
            ' '.join(map(str, x))
            for x in characters
            if x[3] > shape[0]//2 and x[2] <= shape[1]/4
        ])
    ]
    characters = sorted([*map(lambda x: [*map(int, x.split())], characters)], key=lambda x: x[0])
    for i in range(1, len(characters)):
        if characters[i][0] - characters[i-1][0] <= characters[i-1][2]:
            characters[i][1] = -1
    characters = [ x for x in characters if x[1] != -1 and x[3] < shape[0] ]
    # print('after filtering:', characters, characters.__len__())

    license_plate2 = cv2.resize(license_plate, (shape[1]*3, shape[0]*3))
    license_plate2 = cv2.cvtColor(license_plate2, cv2.COLOR_GRAY2BGR)
    
    images = []

    for x, y, w, h in characters:
        images.append(list(np.array(license_plate[y-1:y+h+1, x-1:x+w+1] > 0, dtype=int)))
        x, y, w, h = np.array([x, y, w, h])*3   #just to view it zoomed 3 times
        cv2.rectangle(license_plate2, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("contours", license_plate2)
    cv2.waitKey(0)

    # print('Shape =', shape)

    for i in range(len(images)):
        # images[i] = cv2.resize(np.array(images[i], np.uint8), (50, 50))
        row, col = len(images[i]), len(images[i][0])
        images[i] = [[0]*col for _ in range((50-row)//2)] + \
            images[i] + [[0]*col for _ in range((50-row+1)//2)]
        for j in range(50):
            images[i][j] = [0]*((50-col)//2) + list(images[i][j]) + [0]*((50-col+1)//2)

            k = 1
            while(k < 50):
                if images[i][j][k-1]:
                    images[i][j][k] = 1
                    k+=1
                k+=1

    # contours = np.array([
    #     [[j, i] for j in range(50) for i in range(50) if img[i][j]]
    #     for img in images
    # ])

    images = np.array(images, np.uint8)

    # for i in range(len(images)):
    #     contours, _ = cv2.findContours(images[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #     cv2.drawContours(images[i], contours[-1], -1, 1, 2)

    #debugging
    # for i, x in enumerate(images):
    #     np.savetxt(stdout, x, fmt='%i')
    #     # cv2.imwrite(f'{i}.jpg', x*255)
    #     print()

    # input("press enter after finishing the changes")
    
    # for i in range(len(images)):
    #     images[i] = cv2.imread(f'{i}.jpg', cv2.IMREAD_GRAYSCALE)
    
    predict(images)

if __name__ == "__main__":
    main()
    # # image = (cv2.imread('4.jpg', cv2.IMREAD_GRAYSCALE)//129).astype('uint8')
    # image = (cv2.imread('t3.png', cv2.IMREAD_GRAYSCALE) < 129).astype('uint8')
    # # image = cv2.resize(image, (50, 50))
    # print('shape=', image.shape)
    # # print(image)
    # # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # # for x in (image):
    
    #     # print()
    # predict(np.array([image]))
