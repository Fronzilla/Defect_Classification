import os

PATH = 'post_process_images_test/'

items = os.listdir(PATH)

for item in items:
    old_file = os.path.join(PATH, item)
    new_file = os.path.join(PATH, item.split('png')[0] + 'jpg')
    os.rename(old_file, new_file)