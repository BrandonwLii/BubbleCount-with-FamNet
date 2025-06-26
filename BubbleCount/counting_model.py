import os
import torch
import cv2
import time
import shutil
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, extract_features
from utils import visualize_output_and_save
import BubbleCount.image_preprocess as image_preprocess
import BubbleCount.csv_helpers as csv_helpers

PATCH_TXT_NAME = "patch_coord.txt"
PATCH_WIDTH = 10
ADJUSTED_DIR_NAME = "adjusted_exemplar"



# Main Pipe
class CountingPipe():
    """ Pipeline for counting objects in images using exemplars from sample images

        An implementation of the FamNet model where the object of interest is defined 
        in the sample images with bounding boxes. The paths are defined in the `args`, 
        which should be updated when needed: 
            "sample_path", "target_path", "output_dir", "model_path"

        Developed by Eric Wang, Nov 2023
    """
    def __init__(self, args=None, num_boxes=4):
        self.result_to_csv = []
        self.args = {
            "sample_path": "Exemplars/",        # "G:/My Drive/ECE MEng Courses/ECE2500/LearningToCountEverything/Exemplars"
            "target_path": "Targets/9_batch/",    # "G:\My Drive\ECE MEng Courses\ECE2500\LearningToCountEverything\Targets\9_batch"
            "output_dir": "/Outputs/",
            "model_path": "./data/pretrainedModels/FamNet_Save1.pth",
        }
        if args is not None:
            self.args.update(args)
        self.b_plot_result = False
        self.batch_name = ""
        self.num_boxes = num_boxes

        if not torch.cuda.is_available():
            self.use_gpu = False
            print("===> Using CPU mode.\n")
        else:
            self.use_gpu = True
            print("===> Using GPU mode.\n")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    def pre_pipe(self):
        """The first stage of the pipeline which prepares the adjusted exemplars"""
        self.pinpoint_target_background()
        try:
            self._adjust_brightness()
        except Exception as e:
            print(f"An error occurred: {e}")

    def pinpoint_target_background(self):
        """Walks thorugh the target directory (data directory)

        Prompts the user to select the background point. Then saves the coordinates
        to a txt in the save directory as the image.
        """
        print("----------------------------------------------------------------")
        print("User to pinpoint background of target batches:")
        for root, _, files in os.walk(self.args["target_path"]):
            for file_name in files:
                if not file_name.lower().endswith(".jpg"):
                    continue

                print(f"In directory: {os.path.basename(root)}")

                self._select_point_and_save(root, file_name)
                print("Done with this folder.\n")
                break # only need one image from each target diretory

    def _select_point_and_save(self, dir, file_name):
        """Prompt user to select a point in the background and save coordinates to txt"""
        img_path = os.path.join(dir, file_name)
        # Check if the file exists
        if not os.path.isfile(img_path):
            print(f"Image file '{img_path}' not found.")
            return

        # Read the image
        img = cv2.imread(img_path)
        clicked = False  # Flag to indicate whether the user has clicked

        def click_event(event, x, y, flags, params):
            nonlocal clicked  # Use nonlocal to modify the flag in the outer scope

            if event == cv2.EVENT_LBUTTONDOWN and not clicked:
                # Convert the point to integer coordinates
                x, y = int(x), int(y)

                # Write the coordinate and image title to "patch_coord.txt"
                txt_output_path = os.path.join(dir, PATCH_TXT_NAME)
                with open(txt_output_path, "w") as file:
                    file.write(f"Image: {file_name}, Coordinate: ({x}, {y})\n")

                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot
                cv2.putText(img, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('image', img)

                # Set the clicked flag to True to exit the image display
                clicked = True

        # Display the image
        cv2.imshow('image', img)

        # Display instructions
        print("Click once on the background of the image to record the point.")

        # Setting mouse handler for the image
        cv2.setMouseCallback('image', click_event)

        while not clicked:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # If the user presses the 'Esc' key, exit
                break

        print(f"Point saved in directory: {os.path.basename(dir)}.")
        time.sleep(1)
        cv2.destroyAllWindows()

    def batch_adjust_brightness(self):
        """Adjust the brightness of the exemplars to match each batch of targets

        Creates unique exemplars based on the brightness of the target batch. Saves
        the adjusted exemplar image to a separate folder within the target directory.

        Note: Make sure the exemplars are in separate folders within the Exemplar directory
        """
        print("----------------------------------------------------------------")
        print("Batch adjusting...\n")
        # First, prompt for exeplar patch coordinates
        for root_exemp, _, files_exemp in os.walk(self.args["sample_path"]):
            # print("--")
            if root_exemp == self.args["sample_path"]:
                # keeping another copy of the exemplars in the exemplar folder as well as
                # the subfolders. This step skips over to the subfolders directly 
                continue
            for file_exemp in files_exemp:
                # there should and must be two files in this directory. One jpg
                # and one txt. The txt is the bounding box of the jpg.
                if not file_exemp.lower().endswith('.jpg'):
                    continue
                print(f"Working with {file_exemp}...")
                image_name = os.path.splitext(file_exemp)[0]
                self._select_point_and_save(root_exemp, file_exemp) # create a patch_coord.txt
                patch_txt_path_exemp = os.path.join(root_exemp, PATCH_TXT_NAME)
                exemplar_img, exemplar_patch = self._get_patch_from_txt(patch_txt_path_exemp)

                # Then go through the target dirs and create the adjusted exemplars
                print("\nGoing to match the exemplar with target image batches..")
                for root_targ, _, files_targ in os.walk(self.args["target_path"]):
                    if os.path.basename(root_targ) == ADJUSTED_DIR_NAME:
                        # to not get into the adjusted folder
                        continue
                    print(f"\nIn target folder {root_targ}")
                    if PATCH_TXT_NAME in files_targ:
                        adjusted_exemplar_dir = os.path.join(root_targ, ADJUSTED_DIR_NAME)
                        # Create the directory if it doesn't exist
                        if not os.path.exists(adjusted_exemplar_dir):
                            os.makedirs(adjusted_exemplar_dir)

                        patch_txt_path_targ = os.path.join(root_targ, PATCH_TXT_NAME)
                        _, target_patch = self._get_patch_from_txt(patch_txt_path_targ)

                        # Adjust the exemplar to match the brightness of th etarget patch
                        adjusted_img = self._adjust_brightness(exemplar_img, exemplar_patch, target_patch)
                        adjusted_img_path = os.path.join(root_targ, f"{ADJUSTED_DIR_NAME}/{file_exemp}")
                        cv2.imwrite(adjusted_img_path, adjusted_img.astype(int))
                        print("Adjusted exemplar saved.")

                        # Copy the bounding box to the adjusted folder
                        source_bbox = os.path.join(root_exemp, f"{image_name}_box.txt")
                        target_bbox = os.path.join(adjusted_exemplar_dir, f"{image_name}_box.txt")
                        shutil.copy(source_bbox, target_bbox)
                        print("Bounding box file copied.")

    def _get_patch_from_txt(self, txt_file_path):
        """Reading the coordinate files and return a image patch

        The coordinate files are created in `_select_point_and_save` which is in
        the form of Image: "image_name", Coordinate: (w, h)
        """
        with open(txt_file_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith("Image:"):
                # Extract the image name from the line
                img_name = line.split("Image: ")[1].split(",")[0].strip()

                # Extract the coordinates from the line and convert to integers
                coordinates = line.split("Coordinate: ")[1].strip().strip('()').split(',')
                w = int(coordinates[0])
                h = int(coordinates[1])

        img_path = os.path.join(os.path.dirname(txt_file_path), img_name)
        img = cv2.imread(img_path,1)
        return img.copy(), img.copy()[h : h+PATCH_WIDTH, w : w+PATCH_WIDTH]

    def _adjust_brightness(self, exemplar_img, exemplar_patch, target_patch):
        """White patch method for white balancing"""
        transformed_exemplar = exemplar_img * target_patch.mean(axis=(0, 1)) / exemplar_patch.mean(axis=(0, 1))
        clipped_transformed_exemplar = transformed_exemplar.clip(0, 255)

        return clipped_transformed_exemplar

    def count_hybrid(self, hybrid, hybrid_boxes, exemplar_name, image_name):
        resnet50_conv = Resnet50FPN()
        regressor = CountRegressor(6, pool='mean')

        # hybrid = Normalize(hybrid)
        boxes = torch.Tensor(hybrid_boxes)
        if self.use_gpu:
            hybrid = hybrid.cuda()
            boxes = boxes.cuda()
            resnet50_conv.cuda()
            regressor.cuda()
            regressor.load_state_dict(torch.load(self.args["model_path"]))
        else:
            regressor.load_state_dict(torch.load(self.args["model_path"], map_location=torch.device('cpu')))

        resnet50_conv.eval()
        regressor.eval()

        # feature extraction function
        with torch.no_grad():
            features = extract_features(resnet50_conv, hybrid.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

        with torch.no_grad(): output = regressor(features)

        count = output.sum().item()

        # Result image and csv
        print(f"===> The predicted count for {image_name} is: {count:6.2f}")

        if self.b_plot_result:
            # plot 1 result for each batch
            rslt_file_name = f"{self.args['output_dir']}{self.batch_name}_{exemplar_name}_out.png"
            visualize_output_and_save(hybrid.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file_name)
            self.b_plot_result = False
            print(f"===> Visualized output of batch{image_name} is saved to {rslt_file_name}")
        return count
    
    def count_hybrid_and_visualize(self, hybrid, hybrid_boxes, exemplar_name, image_name, output_directory):
        resnet50_conv = Resnet50FPN()
        regressor = CountRegressor(6, pool='mean')

        # hybrid = Normalize(hybrid)
        boxes = torch.Tensor(hybrid_boxes)
        if self.use_gpu:
            hybrid = hybrid.cuda()
            boxes = boxes.cuda()
            resnet50_conv.cuda()
            regressor.cuda()
            regressor.load_state_dict(torch.load(self.args["model_path"]))
        else:
            regressor.load_state_dict(torch.load(self.args["model_path"], map_location=torch.device('cpu')))

        resnet50_conv.eval()
        regressor.eval()

        # feature extraction function
        with torch.no_grad():
            features = extract_features(resnet50_conv, hybrid.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

        with torch.no_grad(): output = regressor(features)

        count = output.sum().item()

        # Result image and csv
        print(f"===> The predicted count for {image_name} is: {count:6.2f}")

        # plot all results
        rslt_file_name = f"{output_directory}{image_name}_out.png"
        visualize_output_and_save(hybrid.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file_name)
        print(f"===> Visualized output of batch{image_name} is saved to {rslt_file_name}")
        return count

    def counting_one_batch(self, current_dir):
        """Creates hybrids and count for each batch of images, using adjusted exemplars"""
        # Load exemplars and targets
        exemplar_folder = os.path.join(current_dir, ADJUSTED_DIR_NAME)
        if not os.path.exists(exemplar_folder):
            assert(False)
            raise RuntimeError("Cannot find adjusted exemplars. Please run `pinpoint_target_background`.")

        Exemplars = image_preprocess.load_exemplars_from_directory(exemplar_folder)
        Targets = image_preprocess.load_target_images_from_directory(current_dir)

        self._count_thru_Exemplars_Targets(Exemplars, Targets)

    def _count_thru_Exemplars_Targets(self, Exemplars, Targets):
        for sample_image in Exemplars:
            self.b_plot_result = True
            for target_image in Targets:

                # Creating the hybrid
                hybrid, hybrid_boxes = image_preprocess.insert_cropped(sample_image['image'], target_image['image'], sample_image['box'], self.num_boxes)

                # counting
                hybrid_count = self.count_hybrid(hybrid, hybrid_boxes, sample_image["file_name"], target_image["file_name"])

                self.result_to_csv.append([sample_image['file_name'], target_image['file_name'], hybrid_count])

    def counting_all_batches(self):
        """Walks thorugh the target directory (data directory) and count
           with adjusted exemplars
        """
        print("----------------------------------------------------------------")
        print("Counting Starts...\n")
        for root, _, files in os.walk(self.args["target_path"]):
            if PATCH_TXT_NAME in files:
                self.batch_name = os.path.basename(root)

                print(f"Counting in directory: {self.batch_name}")
                self.counting_one_batch(root)

                print("Done with this directory.\n")
        output_path = os.path.join(self.args["output_dir"], "Output_adjusted.csv")
        csv_helpers.backup_and_clear_csv(output_path)
        csv_helpers.save_to_csv(self.result_to_csv, output_path)
        print("Counting with adjusted exemplars is complete.")

    def counting_all_unadjusted(self):
        """
        Note: make sure the exemplars are together in the exemplar directory before running this
        """
        print("----------------------------------------------------------------")
        print("Counting without adjusting exemplars...\n")

        Exemplars = image_preprocess.load_exemplars_from_directory(self.args["sample_path"])

        for root, _, files in os.walk(self.args["target_path"]):
            if os.path.basename(root) == ADJUSTED_DIR_NAME:
                continue
            # if any of the files is a jpg:
            if any(file.lower().endswith(".jpg") for file in files):
                self.batch_name = os.path.basename(root)
                print(f"Counting in directory: {self.batch_name}")

                Targets = image_preprocess.load_target_images_from_directory(root)
                self._count_thru_Exemplars_Targets(Exemplars, Targets)
                print("Done with this directory.\n")
        output_path = os.path.join(self.args["output_dir"], f"Output_unadjusted_set{self.num_boxes}.csv")
        csv_helpers.backup_and_clear_csv(output_path)
        csv_helpers.save_to_csv(self.result_to_csv, output_path)
        print("Counting with unadjusted exemplars is complete.")

    def clean_up(self):
        """Removing all adjusted exemplar folders and patch_coord.txt that were created.

        This is a helper function, mostly used when debuging"""
        for root, dirs, files in os.walk(self.args["target_path"]):
            # Check if any directory in 'dirs' is named "adjusted_exemplar"
            if ADJUSTED_DIR_NAME in dirs:
                print(f"In directory {root}")
                # Construct the full path of the directory to be deleted
                dir_to_delete = os.path.join(root, "adjusted_exemplar")

                # Delete the directory
                try:
                    shutil.rmtree(dir_to_delete)
                    print(f"Deleted directory: {dir_to_delete}")
                except OSError as e:
                    print(f"Error deleting directory {dir_to_delete}: {e}")

            if PATCH_TXT_NAME in files:
                print(f"In directory {root}")
                # Construct the full path of the file to be deleted
                file_to_delete = os.path.join(root, PATCH_TXT_NAME)

                # Delete the file
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted file: {file_to_delete}")
                except OSError as e:
                    print(f"Error deleting file {file_to_delete}: {e}")

