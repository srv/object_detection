#! /usr/bin/env python
import sys
import subprocess
import os
import ConfigParser

class Experiment:
    pass

class CalibrationSet:
    pass

class ImagePair:
    """An image pair"""
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def numString(self):
        left_name = self.left[self.left.rfind("left"):]
        return left_name[4:left_name.rfind(".")]

def collectCalibrationSet(folder_path):
    """Collects calibration files"""
    calib_folder = folder_path + "/calibration"
    calib_left = calib_folder + "/calibration_left.yaml"
    calib_right = calib_folder + "/calibration_right.yaml"
    # try to open files (will raise an error on failure)
    open(calib_left)
    open(calib_right)
    calib_set = CalibrationSet
    calib_set.left = calib_left
    calib_set.right = calib_right
    return calib_set

def collectImagePairs(folder_path):
    """Collects filenames of image pairs"""
    image_folder = folder_path + "/images"
    files = [ f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder,f)) ]
    left_images = [ f for f in files if f.startswith('left') ]
    left_images.sort()
    right_images = [ f.replace('left', 'right') for f in left_images ]
    left_images = [ os.path.join(image_folder, f) for f in left_images ]
    right_images = [ os.path.join(image_folder, f) for f in right_images ]
    pairs = zip(left_images, right_images)
    image_pairs = [ ImagePair(l, r) for (l, r) in pairs ]
    print "found image pairs:"
    for i in image_pairs:
        print i.left, i.right
    return image_pairs
        
def extractAll(folder_path, config):
    calibration_set = collectCalibrationSet(folder_path)
    image_pairs = collectImagePairs(folder_path)
    for image_pair in image_pairs:
        points_file = folder_path + "/out/point_clouds/points" + image_pair.numString() + ".pcd"
        descriptors_file = folder_path + "/out/point_clouds/descriptors" + image_pair.numString() + ".pcd"
        cmd = ["rosrun", "stereo_feature_extraction", "extractor"]
        cmd.append("-L")
        cmd.append(image_pair.left)
        cmd.append("-R")
        cmd.append(image_pair.right)
        cmd.append("-J")
        cmd.append(calibration_set.left)
        cmd.append("-K")
        cmd.append(calibration_set.right)
        cmd.append("-C")
        cmd.append(points_file)
        cmd.append("-D")
        cmd.append(descriptors_file)
        cmd.append("-V")
        for (k, v) in config.items("extractor"):
            cmd.append("--" + k)
            cmd.append(v)
        print "Running extractor..."
        print " ".join(cmd)
        if subprocess.call(cmd) != 0:
            print "ERROR running extractor!"
            sys.exit(2)

def main(argv):
    if len(argv) != 3:
        print >>sys.stderr, "Usage: {0} <test data folder> <config file>".format(argv[0])
        return 1

    folder_path = os.path.normpath(argv[1])

    if not os.path.isdir(folder_path):
        print >>sys.stderr, "ERROR: Path {0} not found!".format(folder_path)
        return 1

    print "Using test data folder {0}".format(folder_path)

    config_file = argv[2]

    if not os.path.isfile(config_file):
        print >>sys.stderr, "ERROR: Config file {0} does not exist, please specify a config file!".format(config_file)
        return 1

    print "Using config file", config_file
    config = ConfigParser.ConfigParser()
    config.readfp(open(config_file))

    # create dirs if necessary
    if not os.path.isdir(folder_path + "/out"):
        os.makedirs(folder_path + "/out")

    if not os.path.isdir(folder_path + "/out/point_clouds"):
        os.makedirs(folder_path + "/out/point_clouds")

    extractAll(folder_path, config)


if __name__ == "__main__":
    sys.exit(main(sys.argv))


