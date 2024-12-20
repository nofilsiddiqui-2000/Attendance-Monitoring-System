# Classroom-Monitoring-System
This system records attendance along with their expression, apart from that it also counts the number of students present in frame using computer vision techniques.
Following are the steps to implement this system:
1. Generate Encodings
  - Create a Folder under the Faces folder with the name of the person. Add atleast 5-10 pictures in that folder.
  - Run the train_v2.py
  - After running it will generate a encodings.pkl file in the encodings folder.
  - Make sure the encodings are generated
2. Attendance Module
  - Copy the encodings.pkl file in the encodings folder of Attendance Module
  - Now you are good to go and run the main file.
  - runfile.py was used to make this system run on Raspberry Pi 3B

3. Rest of files were used for training the FER Module

![Pic1](https://github.com/MaazK7/Classroom-Monitoring-System/assets/115479920/32e237a3-96c5-4455-9228-2c0d12fe8c9d)
