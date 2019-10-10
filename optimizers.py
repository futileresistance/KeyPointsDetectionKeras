from keras.optimizers import Adam

def schedule(epoch, lr):
    if epoch < 3:
        return 4e-4
    elif epoch < 8:
        return 8e-5
    else:
        return 4e-5
