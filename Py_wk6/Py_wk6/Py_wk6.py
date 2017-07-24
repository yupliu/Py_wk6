#Alan week6

import graphlab
image_train = graphlab.SFrame('D:\ML_Learning\\image_train_data\\')
image_test = graphlab.SFrame('D:\\ML_Learning\\image_test_data\\')

image_train['label'].sketch_summary()

img_dog = image_train[image_train['label']=='dog']
img_cat = image_train[image_train['label']=='cat']
img_bird = image_train[image_train['label']=='bird']
img_automobile = image_train[image_train['label']=='automobile']

dog_model = graphlab.nearest_neighbors.create(img_dog,features=['deep_features'],label='id')
cat_model = graphlab.nearest_neighbors.create(img_cat,features=['deep_features'],label='id')
bird_model = graphlab.nearest_neighbors.create(img_bird,features=['deep_features'],label='id')
automobile_model = graphlab.nearest_neighbors.create(img_automobile,features=['deep_features'],label='id')

rat = image_test[0:1]

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')

dog_show = lambda i: get_images_from_ids(dog_model.query(image_test[i:i+1]))['image'].show()
cat_show = lambda i: get_images_from_ids(cat_model.query(image_test[i:i+1]))['image'].show()


cat_rat = cat_model.query(rat)
dog_rat = dog_model.query(rat)
cat_rat['distance'].mean()
dog_rat['distance'].mean()



#graphlab.canvas.set_target('ipynb')