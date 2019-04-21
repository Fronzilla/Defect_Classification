import shutil
import os
import random

source = r'C:\Users\av.nikitin\Desktop\docx-image-extractor-master\docx-image-extractor\images'
target = r'C:\Users\av.nikitin\Desktop\docx-image-extractor-master\docx-image-extractor\post_process_images'


def process():
    for root, dir, files in os.walk(source):
        for file in files:
            target_path = os.path.join(root, file)
            new_file = (target_path.split('.png')[0] + str(random.randint(1, 10000000))) + '.png'
            os.rename(target_path, new_file)
            shutil.move(new_file, target)

if __name__ == '__main__':
    process()

