from PIL import Image, ImageDraw
import os
import copy
from tqdm import tqdm
from utils import Transform


def load_exemplars_from_directory(dir_path, reverse_bbox=False):
    """Load images from a directory along with their corresponding bounding boxes.

    Args:
        target_path: string, path of directory that contains the sample images and bounding boxes
        reverse_bbox: to reverse the order of the boxes. This is mainly used when experimenting
                      with number of exemplars, in order to pick different sets of exemplars.

    Return:
        Exemplars: a list of dictionaries of the format {"file_name": file_name, "image": image,  'box': bboxes}
    """
    Exemplars = []

    for file_name in os.listdir(dir_path):
        if file_name.lower().endswith(".jpg"):
            image_name = os.path.splitext(file_name)[0]
            image_path = os.path.join(dir_path, file_name)
            image = Image.open(image_path).convert("RGB")
            image.load()

            # Read the corresponding txt file and extract bounding box data
            txt_file_path = os.path.join(dir_path, f"{image_name}_box.txt")
            bboxes = []

            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r") as txt_file:
                    for line in txt_file:
                        bbox = [int(num) for num in line.strip().split()]
                        bboxes.append(bbox)

            if reverse_bbox:
                bboxes = bboxes[::-1]
            Exemplars.append({"file_name": image_name, "image": image, "box": bboxes})

    return Exemplars


def load_target_images_from_directory(target_path):
    """Load images from a directory to a list of dicts

    Args:
        target_path: string, path of directory that contains the target images

    Return:
        Targets: a list of dictionaries of the format {"file_name": file_name, "image": image}
    """

    Targets = []
    for file_name in os.listdir(target_path):
        if file_name.lower().endswith(".jpg"):
            image_name = os.path.splitext(file_name)[0]
            image_path = os.path.join(target_path, file_name)
            image = Image.open(image_path).convert("RGB")
            image.load()
            Targets.append({"file_name": file_name, "image": image})

    return Targets


def crop_images(image_path1, image_path2, crop_size, crop_position, output_path):
    """
    Crop images to the same sizes. Used for running quick tests on the demo. Might
    not be useful anymore
    """
    # Open the input images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Crop the images
    cropped_image1 = image1.crop(
        (
            crop_position[0],
            crop_position[1],
            crop_position[0] + crop_size[0],
            crop_position[1] + crop_size[1],
        )
    )
    cropped_image2 = image2.crop(
        (
            crop_position[0],
            crop_position[1],
            crop_position[0] + crop_size[0],
            crop_position[1] + crop_size[1],
        )
    )

    # Save the cropped images
    cropped_image1.save(output_path + "/cropped_image1.jpg")
    cropped_image2.save(output_path + "/cropped_image2.jpg")

    print("Images cropped and saved successfully.")
    return


def SS_crop_to_interest(image_dir, region, output_dir):
    """Crop the bubble experiment images to the region of interest

    It takes the directory path of the images to crop as well as the output directory
    as inputs. Current region is taken as [65, 770, 1090, 970] (Aug 13)
    Developed this with the experiment photos in mind. The size of those are
    1920 x 1200.

    Args:
        input_img: loaded image with PIL
        region: y1, x1, y2, x2 coordinates of the region (which makes no senes but
                this is established in the repo, and I don't want to make it consfusing.)
        output_dir: string of output directory

    Output: a cropped image
    """
    # Extract the region coordinates
    y1, x1, y2, x2 = region
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(".jpg"):
            continue
        # Crop the image using the region coordinates
        image_to_crop = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
        image_to_crop.load()
        # print(f"Image {image_name} loaded from {image_dir}.")

        cropped_image = image_to_crop.crop((x1, y1, x2, y2))

        # Save the cropped image to the specified directory
        cropped_path = os.path.join(
            output_dir, image_name.replace(".jpg", "_cropped.jpg")
        )
        cropped_image.save(cropped_path)

    print(f"Cropped images in {image_dir}")
    return


def crop_to_interest(image_dir, region, output_dir):
    """Crop the bubble experiment images to the region of interest

    It takes the directory path of the images to crop as well as the output directory
    as inputs. Current region is taken as [65, 770, 1090, 970] (Aug 13)
    Developed this with the experiment photos in mind. The size of those are
    1920 x 1200.

    Args:
        input_img: loaded image with PIL
        region: y1, x1, y2, x2 coordinates of the region (which makes no senes but
                this is established in the repo, and I don't want to make it consfusing.)
        output_dir: string of output directory

    Output: a cropped image
    """
    # Extract the region coordinates
    y1, x1, y2, x2 = region
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    batch_name = image_dir.split("\\")[-1]

    for image_name in tqdm(os.listdir(image_dir), desc="Processing Images"):
        if not image_name.lower().endswith(".jpg"):
            continue
        # Crop the image using the region coordinates
        image_to_crop = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
        image_to_crop.load()

        cropped_image = image_to_crop.crop((x1, y1, x2, y2))

        # Save the cropped image to the specified directory
        cropped_path = os.path.join(
            output_dir, image_name.replace(".jpg", "_cropped.jpg")
        )
        cropped_image.save(cropped_path)

    print(f"Cropped images in {batch_name}")
    return


def insert_cropped(sample_img, target_img, boxes, num_box=4):
    """Take the bounded exemplar and paste in the target image.
        It supports max of 6 boxes, with a default of 4.

    Args:
        sameple_img: a jpg loaded in with PIL. The image from which the exemplars
                     are taken.
        target_img: a jpg loaded in with PIL. The image whose bubble count is of
                    interests.
        boxes: a nested lists for the coordinates of the bounding boxes.

    Return:
        target_img: with the inserted exemplars
        inserted_positions: new coordinates of the bounding boxes for the inserted exemplars
    """

    # Verify the number of boxes is not more than 6
    # For future use, if it is REALLY desired, update this function to enable more boxes
    if len(boxes) > 6:
        raise ValueError("Exceeded maximum number of boxes (6)")

    hybrid_img = copy.deepcopy(target_img)  # So that the original is not modified
    inserted_positions = []
    # Iterate over the boxes
    for i, box in enumerate(boxes):
        # Verify the number of coordinates is correct
        if len(box) != 4:
            raise ValueError(f"Invalid box coordinates provided for box {i + 1}")

        if i >= num_box:
            # user gets to choose at test time of how many boxes from the exemplar image to include
            break

        y1, x1, y2, x2 = box
        cropped_section = sample_img.crop((x1, y1, x2, y2))

        # Calculate the corner position in the target image
        if i == 0:
            position = (
                hybrid_img.width - cropped_section.width,
                hybrid_img.height - cropped_section.height,
            )  # Bottom right corner
        elif i == 1:
            position = (hybrid_img.width - cropped_section.width, 0)  # Top right corner
        elif i == 2:
            position = (
                0,
                hybrid_img.height - cropped_section.height,
            )  # Bottom left corner
        elif i == 3:
            position = (0, 0)  # Top left corner
        elif i == 4:
            position = (0, hybrid_img.height // 2)  # Middle of left edge
        elif i == 5:
            position = (
                hybrid_img.width - cropped_section.width,
                hybrid_img.height // 2,
            )

        # Paste the cropped section into the target image
        hybrid_img.paste(cropped_section, position)

        inserted_position = [
            position[1],
            position[0],
            position[1] + cropped_section.height,
            position[0] + cropped_section.width,
        ]
        inserted_positions.append(inserted_position)

    sample = {"image": hybrid_img, "lines_boxes": inserted_positions}
    sample = Transform(sample)
    # Save the modified target image
    return sample["image"], sample["boxes"]


def draw_rectangles_on_image(image_path, coordinates_file):
    image = Image.open(image_path)

    # Open the coordinates file
    with open(coordinates_file, "r") as file:
        lines = file.readlines()

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Process each line in the coordinates file
    for n, line in enumerate(lines):
        y1, x1, y2, x2 = map(int, line.strip().split())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x2 + 2, y1 - 1), str(n + 1), fill="red")

    display(image)
