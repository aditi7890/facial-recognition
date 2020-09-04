import numpy as np 
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
from sklearn.externals import joblib
train_data=np.zeros((7*41,280))
train_label=np.zeros((7*41))
count=-1
#plt.figure(1)
#plt.ion()
# feature extraction 
for i in range(1,42):
    for j in range(1,8):
        plt.cla()
        count=count+1
        path = './orl_face/orl_face/u%d/%d.png'%(i,j)
        im = mimg.imread(path)
        feat,hog_image = feature.hog(im,orientations=8,pixels_per_cell=(16,16),
                                     visualize=True,block_norm='L2-Hys',
                                     cells_per_block=(1,1))
        train_data[count,:]=feat.reshape(1,-1)
        train_label[count]=i
        #plt.subplot(2,1,1)
        #plt.imshow(im,cmap='gray')
        #plt.subplot(2,1,2)
        #plt.imshow(hog_image,cmap='gray')
        #plt.pause(0.3)
        print(i,j)

# model creation
svm_model = svm.SVC(kernel='poly',gamma='scale')

# train the model
svm_model = svm_model.fit(train_data,train_label)

joblib.dump(svm_model,'svm_face_train_modelnew.pkl')

print('training done ')



