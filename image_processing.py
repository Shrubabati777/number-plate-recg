from sys import stdout, exit
import os
try:    #if numpy is not found
    import numpy as np
except Exception as e:
    print(e)
    print("Install numpy with 'pip3 install numpy' command (or pip)")
    exit(1)

try:  # if cv2 is not found
    import cv2
except Exception as e:
    print(e)
    exit(1)

np.set_printoptions(threshold=np.inf) #enable this while debugging

DATA_ROOT = '../English/Fnt/'
arr = []

def main():
    global arr

    for i, d in enumerate(os.listdir(DATA_ROOT)[:36]):
        print(d)
        for image in os.listdir(f'{DATA_ROOT}{d}/'):
            img = cv2.imread(f'{DATA_ROOT}{d}/{image}', cv2.IMREAD_GRAYSCALE)
            # img.show()
            img = cv2.resize(img, (50, 50))
            a = cv2.Canny(img, 120, 200)//129  #detecting the edges
            # print(image, '>', a.shape)
            # np.savetxt(stdout, a, fmt='%i')
            # print()
            arr.append(a)   #appending to the list
        print('Processing', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i], 'done')  # verbose


    # # printing
    # for x in arr:
    #     np.savetxt(stdout, x, fmt='%i')
    #     print()

    arr = np.array(arr)
    arr = arr.reshape((arr.shape[0], -1))
    # print(arr)
    np.save('processed_images100.npy', arr)    #creating the final pocessed data

# print(os.listdir('../English/Fnt/Sample001'), sep='\n') #debugging

if __name__ == "__main__":
    main()
    print('Done processing! Shape =', arr.shape)

# # debugging
# images = np.load('processed_images.npy', allow_pickle=True)
# print(images.shape)
