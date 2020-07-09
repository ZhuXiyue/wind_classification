import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

unitlen=112
## get data from cuts 
x_paths = glob.glob("./cutx_112/*.npy")
y_paths = glob.glob("./cuty_112/*.npy")
# x_paths = np.load("./cutx_unit16/*.npy")
# y_paths = np.load("./cuty_unit16/*.npy")
#print(x_paths)
x_paths.sort()
y_paths.sort()
len_all = len(x_paths)

batch_num = 100
begin, end = 0, batch_num

while begin <= len_all:
    print("!!!!!!!!!!",begin,end,"!!!!!!!!!")
    if end <= len_all:
        curx_paths = x_paths[begin:end]
        cury_paths = y_paths[begin:end]
    else:
        curx_paths = x_paths[begin:]
        cury_paths = y_paths[begin:]

    datax = []
    datay = []
    for i in range(len(curx_paths)):
        try:
            cur_x = np.load(curx_paths[i])
            cur_y = np.load(cury_paths[i])
            # print("read data from ",cury_paths[i])
        except:
            print("warning!!:cannot open file: "+cury_paths[i])
            continue
        datay.append(cur_y)
        datax.append(cur_x)


    datax = np.array(datax)
    datay = np.array(datay)
    #print(datay[0,0:10,0:100])
    #print(type(datay[0][0][0]))
    
    final_datax = []
    final_datay = []
    for i in range(len(datax)):
        if np.sum(datay[i]>0.5) >= 0.2 * 112*112:
            final_datax.append(datax[i])
            #print("datax",datax[i])
            final_datay.append(datay[i])
    
    final_datax = np.array(final_datax)
    final_datay = np.array(final_datay)

    print(np.shape(final_datax))
    print(np.shape(final_datay))
    # plt.figure()
    # plt.imshow(final_datax[0][:,:,0])
    # plt.figure()
    # plt.imshow(final_datay[0][:,:,0])
    ########3 resize
    resized_final_datax= []
    for i in range(len(final_datax)):
        temp_array = []
        for j in range(np.shape(final_datax)[-1]):  
            temp = Image.fromarray(final_datax[i,:,:,j])
            temp = temp.convert("I")
            #print(temp)
            temp = temp.resize((unitlen,unitlen))
            temp = np.array(temp)
            temp_array.append(temp)
        temp_array = np.array(temp_array)
        temp_array = temp_array.transpose(1,2,0)
        #print(np.shape(temp_array))
        resized_final_datax.append(temp_array) 
        
    resized_final_datay= []
    print(type(final_datay[0][0][0]))
    for i in range(len(final_datax)):
        temp = Image.fromarray(final_datay[i,:])
        # temp = temp.convert("I")
        temp = temp.resize((unitlen,unitlen))
        temp = np.array(temp)
        resized_final_datay.append(temp) 
    resized_final_datax = np.array(resized_final_datax) 
    resized_final_datay = np.array(resized_final_datay)
    print(np.shape(resized_final_datax))
    print(np.shape(resized_final_datay))
    #print(resized_final_datay[0,0:10,0:100])
    print("!@#@$%",np.sum(resized_final_datay)/np.shape(resized_final_datay)[0]/112/112)      


    trainx,testx, trainy, testy = train_test_split(resized_final_datax,resized_final_datay,test_size=0.2)
    trainx = np.array(trainx)
    testx = np.array(testx)
    trainy = np.array(trainy)
    trainy = trainy.reshape((len(trainy),unitlen,unitlen,1))
    testy = np.array(testy)
    testy = testy.reshape((len(testy),unitlen,unitlen,1))

    # for i in range(len(testx)):
    #     testx[i] = testx[i]/10000
    # for i in range(len(trainx)):
    #     trainx[i] = trainx[i]/10000
    #print(trainy[0,0:100,0:10,0])


    print("max_val in x:", np.max(trainx))
    trainx = trainx/255.
    testx = testx/255.
    print("finally! trainx shape:",np.shape(trainx))
    print("finally! testx shape:",np.shape(testx))
    print("finally! trainy shape:",np.shape(trainy))
    print("finally! testy shape:",np.shape(testy))

    for i in range(len(trainx)):
        np.save("./split_data_54/trainx/trainx"+str(i+begin)+".npy",trainx[i])
        np.save("./split_data_54/trainy/trainy"+str(i+begin)+".npy",trainy[i])
    for i in range(len(testx)):
        np.save("./split_data_54/testx/testx"+str(i+begin)+".npy",testx[i])
        np.save("./split_data_54/testy/testy"+str(i+begin)+".npy",testy[i])

    ## print the zero ratio
    # print(trainy[0])
    zero_ratio = 1 - np.sum(trainy)/(np.shape(trainy)[0]*np.shape(trainy)[1]*np.shape(trainy)[2]*np.shape(trainy)[3])
    print("zero ratio:",zero_ratio)
    del datax
    del datay
    del final_datax
    del final_datay
    del resized_final_datax
    del resized_final_datay

    begin, end = end, end +batch_num
