import os
import cv2
from pathlib import Path
import argparse
import random

# -> root_image_dir
#   -> 1.jpg
#   -> 2.jpg
#   -> ...
#   -> occlusions
#       -> occlusions_1
#       -> occlusions_2
#           -> 1.jpg
#           -> ...

occlusion_size_factor = 5

def draw_random_square(image):
    height, width, _ = image.shape

    # Draw a rectangle
    margin = height / 10
    x = random.randint(margin, width-margin)
    y = random.randint(margin, height-margin)
    w = h = random.randint(int(margin/occlusion_size_factor), int(3*margin/occlusion_size_factor))
    color = random.choice([
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=-1)
    return image

def draw_random_circle(image):
    height, width, _ = image.shape

    # Draw a rectangle
    margin = height / 10
    x = random.randint(margin, width-margin)
    y = random.randint(margin, height-margin)
    r = random.randint(int(margin/occlusion_size_factor), int(3*margin/occlusion_size_factor))
    color = random.choice([
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ])
    cv2.circle(image, (x, y), r, color, thickness=-1)
    return image

def main(data_dir, occlusion_level):
    assert "occlusions" not in data_dir, "Better to not operate on nested occlusion dirs"
    assert occlusion_level > 0, "Occlusion level must be positive integer"
    root_image_dir = Path(data_dir)
    occlusions_dir = root_image_dir / f"synthetic_occlusions"


    if not occlusions_dir.exists():
        occlusions_dir.mkdir()

    existing_occlusion_subdirs = os.listdir(occlusions_dir)
    if existing_occlusion_subdirs:
        last_occlusion_level = max([int(x.split("_")[-1]) for x in existing_occlusion_subdirs])
        last_occlusion_subdir = occlusions_dir / f"occlusions_{last_occlusion_level}"
    else:
        last_occlusion_level = 0
        last_occlusion_subdir = root_image_dir
    
    assert last_occlusion_level < occlusion_level, "This occlusion level already exists"


    for i in range(last_occlusion_level+1, occlusion_level+1):
        image_names = [x for x in os.listdir(last_occlusion_subdir) if '.jpg' in x]
        current_occlusion_subdir = occlusions_dir / f"occlusions_{i}"
        if not current_occlusion_subdir.exists():
            current_occlusion_subdir.mkdir()

        for image_name in image_names:
            in_path = last_occlusion_subdir / image_name
            image = cv2.imread(str(in_path))
            occlude = random.choice([draw_random_square, draw_random_circle])
            occluded_image = occlude(image)
            out_path = current_occlusion_subdir / image_name
            cv2.imwrite(str(out_path), occluded_image)
            
        
        ...

        last_occlusion_subdir = current_occlusion_subdir

    # # Load the image
    # image = cv2.imread(str(root_image_dir / os.listdir(root_image_dir)[-1]))
    # # Show the image
    # occlusion = random.choice([draw_random_circle, draw_random_square])
    # image = occlusion(image)
    # cv2.imshow("Image with occlusion", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, help='non occluded image path to operate on', required=True)
    parser.add_argument('-o', '--occlusion-level', type=int, help='occlusion level (number of times to occlude an image)', required=True)
    kwargs = vars(parser.parse_args())
    main(**kwargs)