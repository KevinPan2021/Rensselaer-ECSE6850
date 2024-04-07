import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
    
class TWO_Stream(tf.keras.Model):
    def __init__(self):
        super(TWO_Stream, self).__init__()
        
        # CNN + MLP
        self.conv1 = layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='VALID', activation = "relu")
        self.b1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(3, strides=2, padding='VALID')
        self.conv2 = layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='VALID', activation = "relu")
        self.b2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool2D(3, strides=2, padding='VALID')
        self.conv3 = layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='VALID', activation = "relu")
        self.b3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPool2D(3, strides=2, padding='VALID')
        
        # CNN + LSTM
        self.LSTM = layers.LSTM(units=128, activation='tanh', return_sequences=True)
        # multilayer perception
        self.f1 = layers.Flatten()
        self.fc = layers.Dense(11)
        self.fc1 = layers.Dense(11)
        
    def forward(self, input):
        # LSTM + CNN
        frames = 30
        x = tf.reshape(input, [-1, 64, 64, 3])
        x = self.conv1(x)
        x = self.b1(x)
        x = self.pool1(x)
        # batch size, 15, 15, 32
        x = self.conv2(x)
        x = self.b2(x)
        # batch size, 7, 7, 32
        x = self.conv3(x)
        x = self.b3(x)
        x = self.pool3(x)
        x = tf.reshape(x, [-1, frames, 64])
        x = self.LSTM(x)
        # batch size, frames, 128
        x = self.f1(x)
        output0 = self.fc(x)
        
        # CNN + MLP layer
        x1 = input[:,1,:,:,:]
        x1 = tf.reshape(x1, [-1, 64, 64, 3])
        # batch size, 64, 64, 3
        x1 = self.conv1(x1)
        x1 = self.b1(x1)
        x1 = self.pool1(x1)
        # batch size, 15, 15, 32
        x1 = self.conv2(x1)
        x1 = self.b2(x1)
        # batch size, 7, 7, 32
        x1 = self.conv3(x1)
        x1 = self.b3(x1)
        x1 = self.pool3(x1)
        # batch size, 1, 1, 64
        x1 = self.f1(x1)
        output1 = self.fc1(x1)

        # Result Fusion
        output = output0 + output1
        return output
            
def compute_loss(predict, Y): 
    batch, numClass = predict.shape
    onehot = np.zeros((batch, numClass))
    onehot[np.arange(batch), Y] = 1
    #print("Y", Y)
    #print("onehot", onehot)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot, logits=predict)
    loss = tf.math.reduce_mean(cross_entropy)
    return loss

def compute_acc(predict, Y):
    predict = np.argmax(predict, axis=1)
    acc = np.mean(predict == Y)
    return acc

def confusion_plot(Model, data, title):
    predicted = []
    true = []
    for X,Y in data:
        result = Model.forward(X)
        predicted.append(np.argmax(result, axis=1))
        true.append(Y)
    predicted = np.concatenate(predicted).ravel()
    true = np.concatenate(true).ravel()
    matrix = confusion_matrix(true, predicted)
    
    df_cm = pd.DataFrame(matrix, index = [i for i in ['0','1','2','3','4','5','6','7','8','9','10']])
    plt.figure(figsize = (10,7))
    plt.title(title)
    sn.heatmap(df_cm, annot=True)
        
def plot(L_train, L_test, MPJPE_train, MPJPE_test):
    # ploting training loss
    iterations = np.arange(len(L_train))
    plt.figure()
    plt.plot(iterations,L_train)
    plt.title('Train Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    
    # ploting testing loss
    plt.figure()
    plt.plot(iterations,L_test)
    plt.title('Test Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    
    # ploting training accuracy
    plt.figure()
    plt.plot(iterations,MPJPE_train)
    plt.title('Train Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    
    # ploting testing accuracy
    plt.figure()
    plt.plot(iterations,MPJPE_test)
    plt.title('Test Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    

'''
driver function
change "saving" to True to train the model
change "saving" to False to load the model and run TA testset
'''

def main():
    saving = False
    
    # parameters
    eta = 0.001 # learning rata
    #maxItr = 10
    maxItr = 15
    batchSize = 30
    
    Model = TWO_Stream()
    if (saving):       
        # loading  data
        file = open('youtube_action_train_data_part1.pkl', 'rb')
        p1_x, p1_y = pickle.load(file)
        file.close()

        file = open('youtube_action_train_data_part2.pkl', 'rb')
        p2_x, p2_y = pickle.load(file)
        file.close()
        
        dataX = np.append(p1_x, p2_x, axis = 0)
        dataY = np.append(p1_y, p2_y)
        np.random.seed(0)
        
        dataSize = dataY.shape[0]
        split_size = dataSize//4*3
        train_indices = np.random.choice(dataSize,split_size, replace=False)
        all_indices = np.arange(dataSize)
        test_indices = np.delete(all_indices, train_indices)
        trainX = dataX[train_indices]
        testX = dataX[test_indices]
        trainY = dataY[train_indices]
        testY = dataY[test_indices]
        trainN = trainY.shape[0]
        testN = testY.shape[0]
        
        train_data = (tf.data.Dataset.from_tensor_slices((trainX, trainY)).batch(batchSize).shuffle(buffer_size=trainN, seed=1))
        train_data = (train_data.map(lambda x, y:(tf.divide(tf.cast(x, tf.float32), 255.0),y)))
        test_data = (tf.data.Dataset.from_tensor_slices((testX, testY)).batch(batchSize))
        test_data = (test_data.map(lambda x, y:(tf.divide(tf.cast(x, tf.float32), 255.0),y)))

        L_train = []
        L_test = []
        Acc_train = []
        Acc_test = []
        
        # training
        optimizer = optimizers.Adam(learning_rate=eta)
        for i in range(maxItr):
            train_tmp_loss = []
            train_tmp_acc = []
            # randomly select a mini-batch
            train = tqdm(train_data)
            for X, Y in train:
                # back prop
                with tf.GradientTape(persistent = True) as T:
                    predict = Model.forward(X)
                    Acc = compute_acc(predict, Y)
                    Loss = compute_loss(predict, Y)
                    print('Epoch:', i+1, 'loss:', Loss.numpy(), ', acc:', Acc)
                model_gradient = T.gradient(Loss, Model.trainable_variables) 
                optimizer.apply_gradients(zip(model_gradient, Model.trainable_variables))
                train_tmp_loss.append(Loss)
                train_tmp_acc.append(Acc)
            # clearning memory
            train = None
            
            L_train.append(tf.reduce_mean(train_tmp_loss))       
            Acc_train.append(tf.reduce_mean(train_tmp_acc))
            
            # testing dataset
            test_tmp_loss = []
            test_tmp_acc = []
            test = tqdm(test_data)
            for X, Y in test:
                predict = Model.forward(X)
                Loss = compute_loss(predict, Y)
                test_tmp_loss.append(Loss)
                test_tmp_acc.append(compute_acc(predict, Y))
            L_test.append(tf.reduce_mean(test_tmp_loss))       
            Acc_test.append(tf.reduce_mean(test_tmp_acc))     
            # clearning memory
            test = None
        
        # plotting
        confusion_plot(Model, train_data, "Train")
        confusion_plot(Model, test_data, "Test")
        plot(L_train, L_test, Acc_train, Acc_test)
        print("final train accuracy:", Acc_train[-1].numpy())
        print("final test accuracy:", Acc_test[-1].numpy())
        Model.save_weights('model_weights.h5')
        
    '''
    readme (important!):
        Change "youtube_action_train_data_part1.pkl" to whatever .pkl
        Model will be loaded and model preformance will be printed
        Uncomment confusion_plot
    '''
    if (not saving):
        file = open('youtube_action_train_data_part1.pkl', 'rb')
        TA_x, TA_y = pickle.load(file)
        file.close()
        TA_data = (tf.data.Dataset.from_tensor_slices((TA_x, TA_y)).batch(batchSize))
        TA_data = (TA_data.map(lambda x, y:(tf.divide(tf.cast(x, tf.float32), 255.0),y)))
        Model.forward(tf.ones(shape=[1,30,64,64,3]))
        Model.built = True
        Model.load_weights('model_weights.h5')
        L_TA, Acc_TA = [], []
        for X, Y in TA_data:
            predict = Model.forward(X)
            Loss = compute_loss(predict, Y)
            L_TA.append(Loss)
            Acc_TA.append(compute_acc(predict, Y))
        #confusion_plot(Model, TA_data, "TA")
        print("TA Loss:", tf.reduce_mean(L_TA).numpy())  
        print("TA Accuracy:", tf.reduce_mean(Acc_TA).numpy())  
        
if __name__ == "__main__":
    main()