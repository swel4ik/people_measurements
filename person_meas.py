from measurements import *
'''
All transitions between points based on empirical testing
'''
class Front:

  def take_points(self, front_resized_peacks, px_to_cm):
    '''
    Take needed key points from resized peacks and px-coef for the front photo 
    '''
    self.left_shol_x_front = front_resized_peacks[5][0] #left shoulder
    self.left_shol_y_front = front_resized_peacks[5][1]
    self.right_shol_x_front = front_resized_peacks[6][0] #right shoulder
    self.right_shol_y_front = front_resized_peacks[6][1]
    self.left_hip_x_front = front_resized_peacks[11][0] #left hip
    self.left_hip_y_front = front_resized_peacks[11][1]
    self.right_hip_x_front = front_resized_peacks[12][0] #right hip
    self.left_knee_front = front_resized_peacks[13][1] #left knee y
    self.px_to_cm = px_to_cm



  def measure_chest(self, front_mask, image_front, rad):
    '''
    Input: 
      front_mask - mask from segmentation model
      image_front - image for the rendering 
      rad - thickness in pixels for circles and line
     Output:
       Chest len in cm (front side)
    '''
    self.Neck_y_front = (self.left_shol_y_front + self.right_shol_y_front) // 2
    self.waist_cent_y = int((self.Neck_y_front + self.left_hip_y_front) // 2) #near waist 
    self.new_chest_y = self.Neck_y_front + int((self.waist_cent_y - self.Neck_y_front) * 0.5)
    self.Neck_x_front = (self.right_shol_x_front + self.left_shol_x_front) // 2 #chest y level 

    left_x_chest_front = left_point(self.new_chest_y, front_mask) #left point 
    right_x_chest_front = right_point(self.new_chest_y, front_mask) #right point

    draw(image_front, left_x_chest_front, right_x_chest_front, self.new_chest_y, rad) #rendering

    self.chest_len_front = right_x_chest_front - left_x_chest_front
    self.chest_len_front = self.chest_len_front * self.px_to_cm #in cm
  


  def measure_waist(self, front_mask, image_front, rad):
    '''
    Input: 
      front_mask - mask from segmentation model
      image_front - image for the rendering 
      rad - thickness in pixels for circles and line
     Output:
       Waist len in cm (front side)
    '''
    self.waist_cent_y += int((self.left_hip_y_front - self.waist_cent_y) * 0.25) 

    left_x_waist_front = left_point(self.waist_cent_y, front_mask) #left point
    right_x_waist_front = right_point(self.waist_cent_y, front_mask) #right point

    draw(image_front, left_x_waist_front, right_x_waist_front, self.waist_cent_y, rad) #rendering

    self.waist_len_front = right_x_waist_front - left_x_waist_front 
    self.waist_len_front = self.waist_len_front * self.px_to_cm #in cm



  def measure_hip(self, front_mask, image_front, rad):
    '''
    Input: 
      front_mask - mask from segmentation model
      image_front - image for the rendering 
      rad - thickness in pixels for circles and line
     Output:
       Hip len in cm (front side)
    '''
    self.hip_x_centr = (self.right_hip_x_front + self.left_hip_x_front) // 2 

    part = (self.left_knee_front - self.left_hip_y_front) // 9
    self.new_hip_y = self.left_hip_y_front + part

    left_x_hip_front = left_point(self.new_hip_y, front_mask) #left point
    right_x_hip_front = right_point(self.new_hip_y, front_mask) #right point

    draw(image_front, left_x_hip_front, right_x_hip_front, self.new_hip_y, rad) #rendering

    self.hip_len_front = right_x_hip_front - left_x_hip_front
    self.hip_len_front = self.hip_len_front * self.px_to_cm #in cm
    
    
class Profile:

  def take_points(self, profile_resized_peacks, px_to_cm):
    '''
    Take needed key points from resized peacks and px-coef for the profile photo 
    '''
    self.left_hip_x_profile = profile_resized_peacks[11][0] #left hip
    self.left_hip_y_profile = profile_resized_peacks[11][1]

    self.left_shol_x_profile = profile_resized_peacks[5][0] #left shoulder
    self.left_shol_y_profile = profile_resized_peacks[5][1]
    self.left_knee_profile = profile_resized_peacks[13][1] #left knee
    self.px_to_cm = px_to_cm

  def measure_chest(self, profile_mask, image_profile, rad):
    '''
    Input: 
      profile_mask - mask from segmentation model
      image_profile - image for the rendering 
      rad - thickness in pixels for circles and line
     Output:
       Chest len in cm (profile side)
    '''
    loc_sho_4 = int((self.left_hip_y_profile - self.left_shol_y_profile) // 4)
    self.y_chest = self.left_shol_y_profile + loc_sho_4
    self.x_chest = self.left_hip_x_profile

    left_x_chest = left_point(self.y_chest, profile_mask) #left point
    right_x_chest = right_point(self.y_chest, profile_mask) #right point

    draw(image_profile, left_x_chest, right_x_chest, self.y_chest, rad) #rendering
    
    self.chest_profile_len = right_x_chest - left_x_chest
    self.chest_profile_len = self.chest_profile_len * self.px_to_cm #in cm

  def measure_waist(self, profile_mask, image_profile, rad):
    '''
    Input: 
      profile_mask - mask from segmentation model
      image_profile - image for the rendering 
      rad - thickness in pixels for circles and line
     Output:
       Waist len in cm (profile side)
    '''
    self.y_talia = int((self.left_hip_y_profile + self.y_chest) // 2)

    left_x_waist = left_point(self.y_talia, profile_mask) #left point
    right_x_waist = right_point(self.y_talia, profile_mask) #right point

    draw(image_profile, left_x_waist, right_x_waist, self.y_talia, rad) #rendering

    self.waist_profile_len = right_x_waist - left_x_waist
    self.waist_profile_len = self.waist_profile_len * self.px_to_cm #in cm

  def measure_hip(self, profile_mask, image_profile, rad):
    '''
    Input: 
      profile_mask - mask from segmentation model
      image_profile - image for the rendering 
      rad - thickness in pixels for circles and line
     Output:
       Hip len in cm (profile side)
    '''

    part = (self.left_knee_profile - self.left_hip_y_profile) // 20

    self.y_hip = self.left_hip_y_profile + part

    left_x_hip = left_point(self.y_hip, profile_mask) #left point
    right_x_hip = right_point(self.y_hip, profile_mask) #right point

    draw(image_profile, left_x_hip, right_x_hip, self.y_hip, rad) #rendering

    self.hip_profile_len = right_x_hip - left_x_hip
    self.hip_profile_len = self.hip_profile_len * self.px_to_cm #in cm


