from flask import Flask, render_template, request, url_for, Response, redirect
import math
import cv2 as cv
import time
import base64
from io import BytesIO
from PIL import Image
from measurements import *
import cv2
from model_class import *
from person_meas import *
app = Flask(__name__)



@app.route('/', methods=['GET'])
def main_page():
    return render_template('entry.html')


@app.route('/calc', methods=['GET', 'POST'])
def calculation():
    real_height = request.form['height_human']
    real_chest = request.form['real_chest']
    real_hip = request.form['real_hip']
    real_waist = request.form['real_waist']

    try:
        real_chest = float(real_chest)
    except ValueError:
        pass
    try:
        real_waist = float(real_waist)
    except ValueError:
        pass
    try:
        real_hip = float(real_hip)
    except ValueError:
        pass

    real_height = float(real_height)
    upload_dir = 'C:/Users/zador/Desktop/roonyx/measure_photo/API/static/'
    front_img = request.files['photo1'].read()
    profile_img = request.files['photo2'].read()
    if front_img and profile_img:

        front_img = np.fromstring(front_img, np.uint8)
        front_img = cv2.imdecode(front_img, cv2.IMREAD_COLOR)
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        front_img = Image.fromarray(front_img.astype("uint8"))
        front_path = 'front.jpg'

        front_img.save(upload_dir + front_path)

        profile_img = np.fromstring(profile_img, np.uint8)
        profile_img = cv2.imdecode(profile_img, cv2.IMREAD_COLOR)
        profile_img = cv2.cvtColor(profile_img, cv2.COLOR_BGR2RGB)
        profile_img = Image.fromarray(profile_img.astype("uint8"))
        profile_path = 'profile.jpg'

        profile_img.save(upload_dir + profile_path)
        image_profile_segm = profile_img.copy()
        image_front_segm = front_img.copy()

        segm_model_path = "C:/Users/zador/Desktop/roonyx/measure_photo/API/models/256_last_model_quant.tflite"
        segm_model = Models(segm_model_path)

        image_front = cv.imread('C:/Users/zador/Desktop/roonyx/measure_photo/API/static/front.jpg')
        image_profile = cv.imread('C:/Users/zador/Desktop/roonyx/measure_photo/API/static/profile.jpg')
        image_front_segm = image_front.copy()
        image_profile_segm = image_profile.copy()

        height_front, width_front, _ = image_front.shape
        height_profile, width_profile, _ = image_profile.shape

        input_front_image = segm_model.preprocessing_image(image_front_segm)
        front_mask = segm_model.get_mask(input_front_image, width_front, height_front)

        input_profile_image = segm_model.preprocessing_image(image_profile_segm)
        profile_mask = segm_model.get_mask(input_profile_image, width_profile, height_profile)
        # cv.imwrite(upload_dir+'image_front.png', image_front_segm)
        # cv.imwrite(upload_dir+'image_profile.png', image_profile_segm)
        # segm_part
        k1 = 200
        # размеры кружков и линий
        rf = image_front.shape[0] // k1
        rp = image_profile.shape[0] // k1
        model_path_posenet = "C:/Users/zador/Desktop/roonyx/measure_photo/API/models/posenet_mv1_075_float_from_checkpoints.tflite"  # 337x337
        posenet_model = Models(model_path_posenet)

        height_front_coef = height_front / posenet_model.trained_image_width
        width_front_coef = width_front / posenet_model.trained_image_width

        height_profile_coef = height_profile / posenet_model.trained_image_width
        width_profile_coef = width_profile / posenet_model.trained_image_width

        front_ratio = [height_front_coef, width_front_coef]
        profile_ratio = [height_profile_coef, width_profile_coef]

        image_front_pose = posenet_model.preprocessing_image(image_front_segm)
        front_peacks, front_scores = posenet_model.get_key_points(image_front_pose)
        front_resized_peacks = posenet_model.resize_peacks(front_peacks, front_ratio)

        image_profile_pose = posenet_model.preprocessing_image(image_profile_segm)
        profile_peacks, profile_scores = posenet_model.get_key_points(image_profile_pose)
        profile_resized_peacks = posenet_model.resize_peacks(profile_peacks, profile_ratio)

        front_points_flag = front_points_check(front_resized_peacks)  # если оба True то фото подходит
        front_pose_flag = front_pose_check(front_resized_peacks, front_scores)
        if (front_points_flag or front_pose_flag) == False:
            return render_template('pose_error.html',
                                   decision='Wrong front pose, try again'), 400

        profile_points_flag = profile_points_check(profile_resized_peacks)
        profile_pose_flag = profile_pose_check(profile_resized_peacks, profile_scores)
        if (profile_points_flag or profile_pose_flag) == False:
            return render_template('pose_error.html',
                                   decision='Wrong front pose, try again'), 400

        try:
            head_front_y = head_front(front_resized_peacks, front_mask)
            heel_front_y = heel(front_resized_peacks, front_mask)
        except IndexError:
            return render_template('pose_error.html',
                                   decision='Something wrong with your front photo, try to change a background or do more quality photo'), 400
        px_to_cm = px_coef(head_front_y, heel_front_y, real_height)

        # front
        person_front = Front()
        person_front.take_points(front_resized_peacks, px_to_cm)
        try:
            person_front.measure_chest(front_mask, image_front, rf)
            person_front.measure_waist(front_mask, image_front, rf)
            person_front.measure_hip(front_mask, image_front, rf)
        except IndexError:
            return render_template('pose_error.html',
                                   decision='Something wrong with your front photo, try to change a background or do more quality photo'), 400

        try:
            head_profile_y = head_profile(profile_resized_peacks, profile_mask)
            heel_profile_y = heel(profile_resized_peacks, profile_mask)
        except IndexError:
            return render_template('pose_error.html',
                                   decision='Something wrong with your profile photo, try to change a background or do more quality photo'), 400
        px_to_cm = px_coef(head_profile_y, heel_profile_y, real_height)

        # profile

        person_profile = Profile()
        person_profile.take_points(profile_resized_peacks, px_to_cm)
        try:
            person_profile.measure_chest(profile_mask, image_profile, rp)
            person_profile.measure_waist(profile_mask, image_profile, rp)
            person_profile.measure_hip(profile_mask, image_profile, rp)
        except IndexError:
            return render_template('pose_error.html',
                                   decision='Something wrong with your profile photo, try to change a background or do more quality photo'), 400

        # image_front = image_front[:,:,::-1] #bgr ---> rgb
        # image_profile = image_profile[:,:,::-1] #bgr ---> rgb

        cv.imwrite('C:/Users/zador/Desktop/roonyx/measure_photo/API/static/image_front.png', image_front)
        cv.imwrite('C:/Users/zador/Desktop/roonyx/measure_photo/API/static/image_profile.png', image_profile)

        key = np.random.randint(0, 5000)

        url = 'image_front.png'
        url_2 = 'image_profile.png'

        CHEST = ellipse(person_front.chest_len_front, person_profile.chest_profile_len)
        WAIST = ellipse(person_front.waist_len_front, person_profile.waist_profile_len)
        HIP = ellipse(person_front.hip_len_front, person_profile.hip_profile_len)
        # final measurements

        try:
            chest_er = round(abs(CHEST - real_chest), 1)
        except TypeError:
            chest_er = '-'
        try:
            waist_er = round(abs(WAIST - real_waist), 1)
        except TypeError:
            waist_er = '-'
        try:
            hip_er = round(abs(HIP - real_hip), 1)
        except TypeError:
            hip_er = '-'

    return render_template('results.html',
                           Chest=CHEST,
                           Waist=WAIST,
                           Hip=HIP,
                           real_chest=real_chest,
                           real_waist=real_waist,
                           real_hip=real_hip,
                           chest_er=chest_er,
                           waist_er=waist_er,
                           hip_er=hip_er,
                           the_title='PEOPLE MEASUREMENT DEMO',
                           url=url,
                           height=len(image_front) // 1.5,
                           width=len(image_front[1]) // 1.5,
                           url_2=url_2,
                           height_2=len(image_profile) // 1.5,
                           width_2=len(image_profile[1]) // 1.5)


@app.route('/search4', methods=['GET', 'POST'])
def do_search():
    data = request.get_json()
    # global real_height
    # global real_height
    # global real_chest
    # global real_waist
    # global real_hip
    real_height = data['height']
    real_chest = data['chest']
    real_waist = data['waist']
    real_hip = data['hip']
    # front_img = data['screenshots'][0]
    # profile_img = data['screenshots'][1]
    front_img = data['screenshots']['first']
    profile_img = data['screenshots']['second']

    real_height = float(real_height)


    try:
        real_chest = float(real_chest)
    except ValueError:
        pass
    try:
        real_waist = float(real_waist)
    except ValueError:
        pass
    try:
        real_hip = float(real_hip)
    except ValueError:
        pass

    starter = front_img.find(',')
    image_1 = front_img[starter+1:]
    image_1 = bytes(image_1, encoding="ascii")
    image_1 = Image.open(BytesIO(base64.b64decode(image_1)))
    image_1 = image_1.convert('RGB')
    image_1.save('static/front.jpg')

    starter_2 = profile_img.find(',')
    image_2 = profile_img[starter+1:]
    image_2 = bytes(image_2, encoding="ascii")
    image_2 = Image.open(BytesIO(base64.b64decode(image_2)))
    image_2 = image_2.convert('RGB')
    image_2.save('static/profile.jpg')
    # return redirect(url_for('/api/model'))
    return redirect("http://127.0.0.1:5000/api/model", code=302)


@app.route('/api/model', methods=['GET', 'POST'])
def model():

    image_front = cv.imread('static/front.jpg')
    image_profile = cv.imread('static/profile.jpg')
    image_front_segm = image_front[:, :, ::-1]
    image_profile_segm = image_profile[:, :, ::-1]
    # except TypeError:
    #     image_front = cv.imread('static/0.png')
    #     image_profile = cv.imread('static/1.png')
    #     image_front_segm = image_front[:, :, ::-1]
    #     image_profile_segm = image_profile[:, :, ::-1]

  #segm_part
    segm_model_path = "models/256_last_model_quant.tflite"
    segm_model = Models(segm_model_path)


    height_front, width_front, _ = image_front.shape
    height_profile, width_profile, _ = image_profile.shape
    
    input_front_image = segm_model.preprocessing_image(image_front_segm)
    front_mask = segm_model.get_mask(input_front_image, width_front, height_front)
    
    input_profile_image = segm_model.preprocessing_image(image_profile_segm)
    profile_mask = segm_model.get_mask(input_profile_image, width_profile, height_profile)

    

  #segm_part
    k1 = 200
  #размеры кружков и линий
    rf = image_front.shape[0] // k1
    rp = image_profile.shape[0] // k1
    model_path_posenet = "models/posenet_mv1_075_float_from_checkpoints.tflite" #337x337
    posenet_model = Models(model_path_posenet)
    
    height_front_coef = height_front / posenet_model.trained_image_width
    width_front_coef = width_front / posenet_model.trained_image_width

    height_profile_coef = height_profile / posenet_model.trained_image_width
    width_profile_coef = width_profile / posenet_model.trained_image_width

    front_ratio = [height_front_coef, width_front_coef]
    profile_ratio =[height_profile_coef, width_profile_coef]
    
    image_front_pose = posenet_model.preprocessing_image(image_front_segm)
    front_peacks, front_scores = posenet_model.get_key_points(image_front_pose)
    front_resized_peacks = posenet_model.resize_peacks(front_peacks, front_ratio)

    image_profile_pose = posenet_model.preprocessing_image(image_profile_segm)
    profile_peacks, profile_scores = posenet_model.get_key_points(image_profile_pose)
    profile_resized_peacks = posenet_model.resize_peacks(profile_peacks, profile_ratio)



    front_points_flag = front_points_check(front_resized_peacks)  # если оба True то фото подходит
    front_pose_flag = front_pose_check(front_resized_peacks, front_scores)
    if (front_points_flag or front_pose_flag) == False:
        return render_template('pose_error.html',
                               decision='Wrong front pose, try again'), 400

    profile_points_flag = profile_points_check(profile_resized_peacks)
    profile_pose_flag = profile_pose_check(profile_resized_peacks, profile_scores)
    if (profile_points_flag or profile_pose_flag) == False:
        return render_template('pose_error.html',
                               decision='Wrong front pose, try again'), 400

    try:
        head_front_y = head_front(front_resized_peacks, front_mask)
        heel_front_y = heel(front_resized_peacks, front_mask)
    except IndexError:
        return render_template('pose_error.html',
                               decision='Something wrong with your front photo, try to change a background or do more quality photo'), 400
    px_to_cm = px_coef(head_front_y, heel_front_y, real_height)


    #front    
    person_front = Front()
    person_front.take_points(front_resized_peacks, px_to_cm)
    try:
        person_front.measure_chest(front_mask, image_front, rf)
        person_front.measure_waist(front_mask, image_front, rf)
        person_front.measure_hip(front_mask, image_front, rf)
    except IndexError:
        return render_template('pose_error.html',
                               decision='Something wrong with your front photo, try to change a background or do more quality photo'), 400
      

    try:
        head_profile_y = head_profile(profile_resized_peacks, profile_mask)
        heel_profile_y = heel(profile_resized_peacks, profile_mask)
    except IndexError:
        return render_template('pose_error.html',
                               decision='Something wrong with your profile photo, try to change a background or do more quality photo'), 400
    px_to_cm = px_coef(head_profile_y, heel_profile_y, real_height)

    #profile 
    
    person_profile = Profile()
    person_profile.take_points(profile_resized_peacks, px_to_cm)
    try:
        person_profile.measure_chest(profile_mask, image_profile, rp)
        person_profile.measure_waist(profile_mask, image_profile, rp)
        person_profile.measure_hip(profile_mask, image_profile, rp)
    except IndexError:
        return render_template('pose_error.html',
                               decision='Something wrong with your profile photo, try to change a background or do more quality photo'), 400





      
      # image_front = image_front[:,:,::-1] #bgr ---> rgb
      # image_profile = image_profile[:,:,::-1] #bgr ---> rgb

    cv.imwrite('static/image_front.png',image_front)
    cv.imwrite('static/image_profile.png',image_profile)
    
    key = np.random.randint(0, 5000)
    
    url = f'static/image_front.png?dummy={key}'
    url_2 = f'static/image_profile.png?dummy={key}'

    CHEST = ellipse(person_front.chest_len_front, person_profile.chest_profile_len)
    WAIST = ellipse(person_front.waist_len_front, person_profile.waist_profile_len)
    HIP = ellipse(person_front.hip_len_front, person_profile.hip_profile_len)
      #final measurements

    try:
        chest_er = round(abs(CHEST - real_chest), 1)
    except TypeError:
        chest_er = '-'
    try:
        waist_er = round(abs(WAIST - real_waist), 1)
    except TypeError:
        waist_er = '-'
    try:
        hip_er = round(abs(HIP - real_hip), 1)
    except TypeError:
        hip_er = '-'
    return render_template('results.html',
                            Chest=CHEST,
                            Waist=WAIST,
                            Hip=HIP,
                            real_chest=real_chest,
                            real_waist=real_waist,
                            real_hip=real_hip,
                            chest_er=chest_er,
                            waist_er=waist_er,
                            hip_er=hip_er,
                            the_title='PEOPLE MEASUREMENT DEMO',
                            url=url,
                            height=len(image_front)//1.5,
                            width=len(image_front[1])//1.5,
                            url_2=url_2,
                            height_2=len(image_profile)//1.5,
                            width_2=len(image_profile[1])//1.5)
  

   
  


  # return "<img src='static/image_front.png'/>"
app.run()
