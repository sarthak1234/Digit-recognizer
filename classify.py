import cv2
import numpy as np
import pickle as pickle
from six.moves import range
pickle_file= "MNIST.pkl"
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  w1 = save['w1']
  w2 = save['w2']
  b1 = save['b1']
  b2 = save['b2']
def softmax(x):
    
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def relu(x):
   return x*(x>0)

def predict(img):
   
   x=img.reshape(1,784)
   x=(x-127.5)/255.0
   layer1_prediction = relu(np.dot(x, w1) + b1)
   final_prediction=np.dot(layer1_prediction,w2)+b2 
   return np.argmax(softmax(final_prediction),1)
  
  





im = cv2.imread("photo_1.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(np.array(roi), (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    number=predict(roi)
    cv2.putText(im, str(int(number[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()    