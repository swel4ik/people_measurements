from measurements import *
	
class Models:
  def __init__(self, model_path):
    '''
    Set the model
    Input:
      model_path - path to model
    '''
    self.interpreter = tf.lite.Interpreter(model_path=model_path)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.trained_image_width = self.input_details[0]['shape'][1] #image shape 

  def preprocessing_image(self, image):
    '''
    Preprocess image for model input 
    '''
    image = cv.resize(image, (self.trained_image_width, self.trained_image_width)) #resize for model train size
    image = np.array(image / 255.0, dtype='float32') 
    image = np.expand_dims(image, axis=0)
    return image

  def get_mask(self, image, orig_width, orig_height):
    '''
    Generate mask from segmentation model
    Input:
      orig_width - width of original image
      orig_height - height of original image
    Output:
      Mask of image - boolean array with original size, threshold 0.5
    '''
    self.interpreter.set_tensor(self.input_details[0]['index'], image) #получение макси для фронта
    self.interpreter.invoke()
    front_mask = self.interpreter.get_tensor(self.output_details[0]['index'])
    front_mask = np.squeeze(front_mask)
    front_mask = cv.resize(front_mask, (orig_width, orig_height))
    front_mask = front_mask > 0.5 #to boolean array
    return front_mask
  
  def get_key_points(self, image):
    '''
    Get key points and confidence scores from Posenet model
    Output:
      See parse_output() in measurements.py
    '''
    self.interpreter.set_tensor(self.input_details[0]['index'], image) 
    self.interpreter.invoke()

    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])#выход модели
    offset_data = self.interpreter.get_tensor(self.output_details[1]['index'])#выход модели

    heatmaps = np.squeeze(output_data)
    offsets = np.squeeze(offset_data)
    peacks, scores = parse_output(heatmaps, offsets, 0.2)
    return peacks, scores

  def resize_peacks(self, peacks, ratio):
    '''
    Return resized key points for original image array
    Input:
      peacks - key points from posenet
      ratio - array [height_front_coef, width_front_coef] - original_shape / trained image shape
    '''
    resized_peacks = np.zeros((17,3),dtype='uint32') #возвращение точек к оригинальному масштабу 
    for i in range(len(peacks)):
      resized_peacks[i,0] = int(peacks[i][1] * ratio[1])
      resized_peacks[i,1] = int(peacks[i][0] * ratio[0])
      resized_peacks[i,2] = peacks[i][2] 
    return resized_peacks