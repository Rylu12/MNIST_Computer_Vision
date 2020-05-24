import tensorflow as tf
import cv2
import numpy as np
from keras import models
import matplotlib.pyplot as plt


#Function for mouse cursor events
def draw_image(event, x, y, flags, param):
    global x1, y1, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (x1, y1), (x, y), (255), 30)

            x1 = x
            y1 = y

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (x1, y1), (x, y), (255), 30)

    return x, y




#Initialize bar plot with all 0s
y_probability = [0,0,0,0,0,0,0,0,0,0]

#Other matplotlib parameters for the bar plot
fig, ax = plt.subplots(figsize=(9,7))
ax.bar(np.arange(0,10), y_probability, label='Prediction Accuracy')
plt.xticks(np.arange(0,10))
plt.ylim((0,100))
plt.xlabel('Numbers Predicted', fontsize=18, weight ='bold')
plt.ylabel('Prediction Probability (%)', fontsize=18, weight ='bold')
plt.rcParams.update({'font.size': 18})

#Ion() and show() function will keep matplotlib running
plt.ion()
plt.show()


#Pre-load the deep learning trained model
#Keras method will do forward propagation through the neural network
model = models.load_model('mnist_DL_pretrained.h5')



drawing = False  # Initialize drawinng variable, becomes True if mouse is pressed



#Initialize a blank 28x28 canvas with all 0s
#28x28 pixel is the input size of images used to train the deep learning model
img = np.zeros((28, 28, 1), np.uint8)

#Resize to 500x500 pixel canvas for more space to draw number
img = cv2.resize(img, (500,500))
clear_img = img.copy()


#Call the mouse event function
cv2.namedWindow('Draw Number')
cv2.setMouseCallback('Draw Number', draw_image)


# Loop will keep running to display the image window along with the bar plot
while True:

    cv2.imshow('Draw Number', img)

    # Copy the most updated image drawing
    image_drawn = img.copy()

    #If something is drawn, pixels will be > value of 1
    #Blank black canvas will be all 0s
    if image_drawn.max()>1:

        # Normalize image pixels so that 255 = 1.0 as max
        image_drawn = image_drawn.astype('float32')/image_drawn.max()

        #Resize image back to 28x28 pixel to feed into model
        image_drawn = cv2.resize(image_drawn, (28, 28))
        image_reshaped = image_drawn.reshape((28,28,1))

        #Model needs a 4-D array with shape (1, 28, 28, 1)
        final_image = np.array([image_reshaped])
        
        prediction = model.predict(final_image)

        #Output predicted values as a list
        y_probability = prediction[0].round(2)*100

        #The below code updates the bar plot in real-time
        plt.cla()
        ax.bar(np.arange(0,10), y_probability, label='Prediction Accuracy')
        plt.xticks(np.arange(0, 10))
        plt.xlabel('Numbers Predicted', fontsize=18, weight ='bold')
        plt.ylabel('Prediction Probability (%)', fontsize=18, weight = 'bold')
        plt.ylim((0, 100))
        plt.rcParams.update({'font.size': 18})
        fig.canvas.draw()


    #10 millisecond delays to save memory
    key = cv2.waitKey(10) & 0xFF

    #Quit program when "q" button is pressed on the keyboard
    if key == ord('q'):
        break

    #Clear the drawing and bar plot if "c" is pressed
    elif key == ord('c'):
        img = np.zeros((28, 28, 1), np.uint8)
        img = cv2.resize(img, (500,500))
        image_drawn = img

        y_probability = [0,0,0,0,0,0,0,0,0,0]
        plt.cla()
        ax.bar(np.arange(0, 10), y_probability, label='Prediction Accuracy')
        plt.xticks(np.arange(0, 10))
        plt.xlabel('Numbers Predicted', fontsize=18, weight ='bold')
        plt.ylabel('Prediction Probability (%)', fontsize=18, weight ='bold')
        plt.ylim((0, 100))
        plt.rcParams.update({'font.size': 18})
        fig.canvas.draw()

#When loop ends, close all popped up windows
cv2.destroyAllWindows()
