import landmark as LMx
import cv2
import os
import numpy as np
import dlib

# face_detector = LMx.Face()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:\\Local Projects\\vid2vid\\shape_predictor_68_face_landmarks.dat')


folder_path = "D:\\Local Projects\\vid2vid\\vid2vid-master\\datasets\\face\\test_img\\0001\\"
dest_path = "D:\\Local Projects\\vid2vid\\vid2vid-master\\datasets\\face\\test_keypoints\\0001\\"
images = os.listdir(folder_path)

#rename files
#we assume its name is ImageXX.jpg


for img_file in images:
    num = str(int(img_file[5:-4])-1)
    if len(num) == 1:
        num = "0" + num
    new_num = "000"+num+".jpg"
    new_name = folder_path+new_num
    os.rename(folder_path+img_file, new_name)
    img = cv2.imread(new_name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = detector(imgRGB, 1)
    for rect in rects:
        # Get the landmark points
        shape = predictor(imgRGB, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

    with open(dest_path+new_num[:-4]+".txt",'w') as out_f:
        for coords in shape:
            out_f.writelines(str(coords[0])+","+str(coords[1])+"\n")



