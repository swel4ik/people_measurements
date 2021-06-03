import tensorflow as tf
import numpy as np
import cv2 as cv
import math
import base64
from io import BytesIO
from PIL import Image

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def parse_output(heatmap_data, offset_data, threshold):  # функция для получение координат точек
  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have low probability
    confidence scores of key points
  '''
  joint_num = heatmap_data.shape[-1]  # number of key points == number of the image's channels
  pose_kps = np.zeros((joint_num, 3), np.uint32)  
  confidence_scores = np.array([])
  for i in range(heatmap_data.shape[-1]):

    joint_heatmap = heatmap_data[..., i]
    max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
    remap_pos = np.array(max_val_pos / 21 * 337, dtype=np.int32)  # 21x21 output size of the image, 337x337 - needed size for the model
    pose_kps[i, 0] = int(remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i])
    pose_kps[i, 1] = int(remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i + joint_num])
    max_prob = np.max(joint_heatmap)
    max_prob = sigmoid(max_prob) #[0,1]
    confidence_scores = np.append(confidence_scores, max_prob) 

    if max_prob > threshold:  
      pose_kps[i, 2] = 1
    else:
      pose_kps[i, :] = 0
  return pose_kps, confidence_scores
  
def left_point(y, maska):
  '''
  Input:
    y - y coordinate
    maska - mask from the segmentation model
  Output:
    Left x coordinate of the mask
  '''
  res = np.min(np.argwhere(maska[y] == True))
  return res

def right_point(y, maska):
  '''
  Input:
    y - y coordinate
    maska - mask from the segmentation model
  Output:
    Left x coordinate of the mask
  '''
  res = np.max(np.argwhere(maska[y] == True)) #
  return res
  

def front_points_check(resized_peacks):
  '''
  Input:
    resized_peacks - resized key points
  Ouput:
    Return True if: needed points were founded (front)
  '''
  #needed keypoints
  if (resized_peacks[0][2]==0 and resized_peacks[1][2]==0) or (resized_peacks[15][2]==0 and resized_peacks[16][2]==0) or (resized_peacks[11][2]==0 and resized_peacks[12][2]==0) or (resized_peacks[5][2]==0 and resized_peacks[6][2]==0):
    #OPTIONALLY: show points which were not found
    # g={0:'nose', 1:'left eye', 15:'left ankle', 16:'right ankle', 5:'left shoulder', 6:'right shoulder', 11:'left hip', 12:'right hip'}
    # miss_points = []
    # for key in g.keys():
    #   if resized_peacks[key][2] == 0:
    #     miss_points.append(g.get(key))
    print('Some points can not be detected')
    needed_points = False
  else: 
    print('All points found')
    needed_points = True

  return needed_points
  
def front_pose_check(resized_peacks, confidence_score):
  '''
  Input:
    resized_peacks - resized key points
    confidence_score - array of confidence scores
  Ouput:
    Return True if: pose is correct (front) (mean confidence score > 0.9 and wrist[y] < nose[y])
  '''
  #pose correction
  if np.mean(confidence_score) > 0.90 and ((resized_peacks[9][1] and resized_peacks[10][1]) < resized_peacks[0][1]): #wrist[y] < nose[y]
    needed_pose = True
    print('Correct pose')
  else: 
    needed_pose = False
    print('Incorrect pose')

  return needed_pose
  
def profile_points_check(resized_peacks):
  '''
  Input:
    resized_peacks - resized key points
  Ouput:
    Return True if: needed points founded (profile)
  '''
  #needed keypoints
  if (resized_peacks[3][2]==0 and resized_peacks[1][2]==0) or (resized_peacks[15][2]==0 and resized_peacks[16][2]==0) or (resized_peacks[11][2]==0 or resized_peacks[5][2]==0 or resized_peacks[13][2]==0):
    print('Some points can not be detected:')
    # d={3:'left ear',1:'left eye',15:'left ankle',16:'right ankle',5:'left shoulder',11:'left hip',13:'left knee'}
    # for key in d.keys():
    #   if resized_peacks[key][2] == 0:
    #     print(d.get(key))
    needed_points = False
  else: 
    print('All points found')
    needed_points = True

  return needed_points
  
def profile_pose_check(resized_peacks, confidence_score):
  '''
  Input:
    resized_peacks - resized key points
    confidence_score - array of confidence scores
  Ouput:
    Return True if: pose is correct (profile) (mean confidence score > 0.7 and < 0.85 and can't be detected right ear)
  '''
  if  0.70 < np.mean(confidence_score) < 0.85 and  resized_peacks[4][2] == 0: #unvisible right ear
    needed_pose = True
    print('Correct pose')
  else:
    needed_pose = False
    print('Incorrect pose')
    
  return needed_pose

  
def head_front(resized_peacks, mask):
  '''
  Input:
    resized_peacks - resized front key points
    mask - mask from segmentation model
  Output:
    Y coordinate of the head
  '''
  if resized_peacks[0][2] != 0:
    high_px_x, high_px_y = resized_peacks[0][:-1] #nose
  else:
    high_px_x, high_px_y = resized_peacks[2][:-1] #right Eye


  while 1:
    if mask[high_px_y][high_px_x] == True:
      high_px_y -= 1
    else: break


  return high_px_y
  
def head_profile(resized_peacks, mask):
  '''
  Input:
    resized_peacks - resized profile key points
    mask - mask from segmentation model
  Output:
    Y coordinate of the head
  '''
  if resized_peacks[3][2] != 0:
    high_px_x, high_px_y = resized_peacks[3][:-1] #nose
  else:
    high_px_x, high_px_y = resized_peacks[1][:-1] #right Eye


  while 1:
    if mask[high_px_y][high_px_x] == True:
      high_px_y -= 1
    else: break


  return high_px_y
 
def heel(resized_peacks, mask):
  '''
  Input:
    resized_peacks - resized key points
    mask - mask from segmentation model
  Output:
    Y coordinate of the heel
  '''
  if resized_peacks[15][2] != 0:
    low_px_x, low_px_y = resized_peacks[15][:-1] #left ankle
  else:
    low_px_x, low_px_y = resized_peacks[16][:-1] #right ankle

  
  while 1:
    if mask[low_px_y][low_px_x] == True:
      low_px_y += 1
    else: break


  return low_px_y

def px_coef(head, heel, real_height):
  '''
  Input:
    head - y coordinate of the head
    heel - y coordinate of the heel
    real_height - person's real height
  Output:
    coefficient - cm in 1 pixel
  '''
  height_in_px = heel - head
  return real_height / height_in_px

def draw(image, left_x_coor, right_x_coor, y_coor, rad):
  '''
  Input:
    image - photo for rendering
    left_x_coor - x coordinate of the left point
    right_x_coor - x coordinate of the right point
    y_coor - y coordinate of the point
    rad - thickness in pixels for circles and line
  '''
  cv.line(image, (left_x_coor, y_coor), (right_x_coor, y_coor), [0,255,0], rad)
  cv.circle(image, (left_x_coor, y_coor), rad, [0,255,0], -1)
  cv.circle(image, (right_x_coor, y_coor), rad, [0,255,0], -1)

def ellipse(a, b):
  '''
  Input:
    a - front lenght
    b - profile lenght
  Output:
    Lenght of ellipse, with which we approximate person's parameter
  '''
  return round(2.0 * math.pi * ((a**2 + b **2) / 8.0)**0.5, 1)


def error(predict_shape, real_shape):
  '''
  Return absolute error (predicted - real), if real shape is empty, return '-'
  '''
  try:
    er = round(abs(predict_shape - real_shape), 1)
  except TypeError:
    er = '-'
  return er

def decode_and_save(img, path):
  '''
  Decode and save image from base64 to jpg
    img - image to decoding
    path - path to saving
  '''
  starter = img.find(',')
  image_1 = img[starter + 1:]
  image_1 = bytes(image_1, encoding="ascii")
  image_1 = Image.open(BytesIO(base64.b64decode(image_1)))
  image_1 = image_1.convert('RGB')
  image_1.save(path)

def encode_to_base64(img_path):
  '''
  Encode image to base64
  '''
  with open(img_path, "rb") as image_file:
    image_base = base64.b64encode(image_file.read())
  image_base = image_base.decode('utf-8')
  return image_base



