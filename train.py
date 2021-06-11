import os

try:  # if numpy is not found
    import numpy as np
except Exception as e:
    print(e)
    print("Install numpy with 'pip3 install numpy' command (or pip)")
    exit(1)

try:  # if PIL is not found
    from PIL import Image
except Exception as e:
    print(e)
    print("Install PIL with 'pip3 install pillow' command (or pip)")
    exit(1)

from recognizer import LogisticRegression
from image_processing import DATA_ROOT

WB = []*36

def accuracy():
    WB = np.array(np.load('trained_data100.npy', allow_pickle=True))
    regressor = [LogisticRegression(0.0001) for _ in range(36)]

    #initialise them with the weights from the trained model
    for i in range(36):
        regressor[i].weights, regressor[i].bias = WB[i]

    acc = []    #stores each prediction
    for i, d in enumerate(os.listdir(DATA_ROOT)[:36]):
        for image in os.listdir(f'{DATA_ROOT}{d}/')[100::9]:
            with Image.open(f'{DATA_ROOT}{d}/{image}') as img:
                # img.show()
                img = img.resize((50, 50))
                img = np.array((np.asarray(img.convert('L'))) < 129,
                            dtype=int)  # making the image b&w
                preds = np.array([regressor[j].predict(img.reshape(-1))
                    for j in range(36)])
                acc.append(i == np.argmax(preds))

        print('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i], 'done') #verbose

    print('Accuracy = ', 100*sum(acc)/len(acc), '%', sep='')    #Accuracy with 100 images = 68% (approx.)

def main():
    X = np.load('processed_images_canny100.npy', allow_pickle=True)  #training data
    y = [[0]*X.shape[0] for _ in range(36)]    #labels
    
    print(X.shape)

    for i in range(36):
        n_trg_imgs_per_chr = X.shape[0]//36 #for 36 classes
        for j in range(n_trg_imgs_per_chr):
            y[i][i*n_trg_imgs_per_chr + j] = 1 #36 labels for 36 regressors for 36 classes

    # for rowzz in y:
    #     print(*rowzz,';')

    print('finished printing labels\n')

    regressor = [LogisticRegression(0.0001) for _ in range(36)]

    for i in range(36):
        regressor[i].fit(X, y[i])   #fitting/training the model
        print('Fitting', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i], 'done')  # verbose

    WB = [[regressor[i].weights, regressor[i].bias] for i in range(36)]
    print(WB)

    WB = np.array(WB)

    np.save('trained_data100.npy', WB)
    print("Saved weights and biasses!")

from time import time

if __name__ == "__main__":
    init = time()
    main()
    print("Final time taken =", time()-init, 'seconds')
    # accuracy()
