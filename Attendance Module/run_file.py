from sklearn.preprocessing import Normalizer
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array 
import cv2
import time
import csv
import time as tp
from datetime import datetime
from scipy.spatial.distance import cosine
import pickle
import numpy as np
import mtcnn
from tflite_runtime.interpreter import Interpreter
from tensorflow import keras

print("Libraries Loaded")





def record_attendance(names, frame,c,emotions):
    # Step 1: Create a dictionary to store the attendance for each date.
    attendance = {}

    # Step 2: Iterate over the names, and check if each name is present in the frame.
    for name in names:
        if name in frame:
            # Step 3: Get the current date and time and add the student to the attendance dictionary.
            now = datetime.now()
            date = now.strftime('%d %b %Y')
            time = now.strftime('%H:%M:%S')
            emotion=emotions[name]
            if date not in attendance:
                attendance[date] = []
            attendance[date].append((name, time,emotion))

    # Print out the frame and names variables for debugging
    print("Frame:", frame)
    print("Names:", names)

    # Step 4: Open the CSV file in append mode to add new rows to the existing data.
    with open('attendance_final{}.csv'.format(c), 'a', newline='') as file:

        # Step 5: Create a csv.writer object.
        writer = csv.writer(file, delimiter=',')

        # Step 6: If the file is empty, write the headers to the CSV file.
        if file.tell() == 0:
            writer.writerow(['Date', 'Name', 'Time', 'Attendance','Emotion'])

        # Step 7: Write the attendance for each date to the CSV file.
        for date in attendance:
            for student in attendance[date]:
                writer.writerow([date, student[0], student[1],student[2], 'Present'])

    # Step 8: Close the CSV file.
    file.close()
    
    
detector = mtcnn.MTCNN()

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')

face_recogn_model = keras.models.load_model('/home/pi/Desktop/Project/model2.h5')


print("Models Loaded")

time_period = 0 #Change this value to increase decrease time period
count=0
# 
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)



def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


recognition_t=0.4
confidence_t=0.99
# 
required_size = (160,160)
emotions={} #MAKE EMpty Dictionary

if __name__ == "__main__":

    required_shape = (160,160)

    face_encoder = face_recogn_model

    #expression=expression_model

    path_m = "facenet_keras_weights.h5"

    face_encoder.load_weights(path_m)
    print("faceeee")

    encodings_path = 'encodings/encodings.pkl'

    encoding_dict = load_pickle(encodings_path)



#### PREDICT USING tflite ###
#tflite with optimization is taking too long on Windows, so not even try.
#On RPi you can try both opt and no opt.

#Load the TFLite model and allocate tensors.
    emotion_interpreter = Interpreter(model_path="/home/pi/Desktop/Emotion/emotion_model__.tflite")
    emotion_interpreter.allocate_tensors()


#Get input and output tensors.
    emotion_input_details = emotion_interpreter.get_input_details()
    emotion_output_details = emotion_interpreter.get_output_details()


#Test the model on input data.
    emotion_input_shape = emotion_input_details[0]['shape']



    class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
    people_present=[]
    

    cap = cv2.VideoCapture(0)

    last_prediction_time = time.time()

    time_period = 2 #Change this value to increase decrease time period






    while cap.isOpened():

        ret,frame = cap.read()

        labels=[]

        if not ret:

            print("CAM NOT OPEND") 

            break

        

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #results=face_classifier.detectMultiScale(gray,1.3,5)
        results = detector.detect_faces(frame)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        name=''

        face_num=len(results)

        for (res) in results:
            if res['confidence'] < confidence_t:
                continue
            x,y,w,h = res['box']
            label_position = (x,y)

            face, pt_1, pt_2 = get_face(img_rgb, (x,y,w,h))

            encode = get_encode(face_encoder, face, required_size)

            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

            name = 'UR'



            distance = float("inf")

            for db_name, db_encode in encoding_dict.items():

                dist = cosine(db_encode, encode)

                if dist < recognition_t and dist < distance:

                    name = db_name

                    distance = dist
                    people_present.append(name)
                    
            color = (0, 255, 0) # green color
            thickness = 2
            scale_factor = 1
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)
            new_x = int(x_center - new_w / 2)
            new_y = int(y_center - new_h / 2)


            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


            roi_gray=gray[y:y+h,x:x+w]

            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            #Get image ready for prediction

            roi=roi_gray.astype('float')/255.0  #Scale

            roi=img_to_array(roi)

            roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)

        
       
            emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
            emotion_interpreter.invoke()
            emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

            label=class_labels[emotion_preds.argmax()]
            emotions[name]=emotion_label #step 2
            label_position=(x,y)
            thickness = 2
            font_scale = 0.5 # adjust this parameter to change the font size
# Ge          t the width and height of the text for string1 and string2
            text1_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text2_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
# Ca          lculate the y-coordinate for the bottom of the first line and the top of the second line
            line1_y = y - 10
            line2_y = line1_y - text1_size[1] - text2_size[1]
# Dr          aw bounding box on image
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
# Ad          d label to image
            cv2.putText(frame, label, (x, line1_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, name, (x, line2_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            label_position=(x+100,y+250)
        record_attendance(['Maaz','Mustafa','Murtaza','Shaheer','Nofil'],people_present,count,emotions)
        label_text = "# of students={}".format(face_num)
        height, width, channels = frame.shape
        # Define the font, scale, color, and thickness of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2
        color = (0, 255, 255)  # white color
        thickness = 2
        # Determine the size of the text string
        size = cv2.getTextSize(label_text, font, scale, thickness)
        # Calculate the coordinates of the bottom-left corner of the text string
        x = int((width - size[0][0]) / 2)  # centered horizontally
        y = height - size[0][1] - 10  # 10 pixels above the bottom edge of the image
        # Draw the text string on the image
        start_time = time.time()
        cv2.putText(frame, label_text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

 
            
        cv2.imshow('camera', frame)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break







