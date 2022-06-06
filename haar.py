import cv2
import random
import time
from scipy.spatial import distance
from skimage.feature import hog
from skimage.transform import resize
from time import sleep

# function calc distance between Matrices Hog(image orignal and cap )
def compare_face(fdi, fd, tolerance=0.5):
    # dis = distance.euclidean(fd, fdi)
    dis =1-distance.cosine(fdi, fd)
    print(f' -  from Fd : {fdi}')
    print(f' -  from Fdi : {fd}')
    print(f' -  from dis : {dis}')
    # with open('attendance.txt', 'r+')as f:
    #     print(f' -  from{fdi}')
    #     f.writelines(f' -  from{fdi}')

    return dis <= tolerance

# detected Face in orignal image ( Haar )
img = cv2.imread("n.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
facesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facese = facesCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces.".format(len(facese)))

for (x, y, w, h) in facese:
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = img[y:y + h, x:x + w]
    # print("[INFO] Object found. Saving locally.")
    # cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi)
# #resize image
# resize image
resized_img = resize(roi, (128, 64))
# generating HOG features
fdi, hog_db = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=True, multichannel=True)


# start_time = time.time()
#--------------
# test cap img in video
#--------------
# vc = cv2.VideoCapture(0)
# while True:
#     ret, image = vc.read()
#
#     # sleep(random.randint(5,10))
#
#     cv2.imwrite('Exam.jpg', image)
#
#     im = cv2.imread("Exam.jpg")
#     if not ret:
#         break
#     gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     facesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     facese = facesCascade.detectMultiScale(
#         gray_im,
#         scaleFactor=1.3,
#         minNeighbors=3,
#         minSize=(30, 30)
#     )
#
#     print("[INFO] Found {0} Faces.".format(len(facese)))
#
#     for (x, y, w, h) in facese:
#         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi_img = im[y:y + h, x:x + w]
#         # print("[INFO] Object found. Saving locally.")
#         # cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi)
#     # #resize image
#     # resize image
#     resized = resize(roi_img, (128, 64))
#     # generating HOG features
#     fd, hogdb = hog(resized, orientations=9, pixels_per_cell=(8, 8),
#                       cells_per_block=(2, 2), visualize=True, multichannel=True)
#
#     # comapare faces
#     results = compare_face(fd,fdi,0.5)
#     print(results)
#
#     cv2.imshow('Exam', im)
#     k = cv2.waitKey(1)
#     if ord('q') == k:
#         break
# vc.release()
# cv2.destroyAllWindows()
# end_time = time.time()
# total_time = end_time - start_time
# print("Time: ", total_time)
#--------------
    # Test just one Image
#--------------
image = cv2.imread("la.png")
gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
facesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facese = facesCascade.detectMultiScale(
    gray_im,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces.".format(len(facese)))

for (x, y, w, h) in facese:
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_img = img[y:y + h, x:x + w]
    # print("[INFO] Object found. Saving locally.")
    # cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi)
# #resize image
# resize image
resized = resize(roi_img, (128, 64))
# generating HOG features
fd, hogdb = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=True, multichannel=True)
results = compare_face(fd,fdi,0.5)
print(results)
