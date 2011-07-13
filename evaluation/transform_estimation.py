#! /usr/bin/env python
import sys
import subprocess
import os

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
        

def collectExperiments(folder_path):
    image_pairs = collectImagePairs(folder_path)
    calibration_set = collectCalibrationSet(folder_path)
    experiments = []
    for index in range(len(image_pairs) - 1):
        from_pair = image_pairs[index]
        to_pair = image_pairs[index + 1]
        transformation_file = folder_path + "/transformations/" + from_pair.numString() + "to" + to_pair.numString() + ".txt"
        print "from", from_pair.left
        print "to", to_pair.right
        print "transformation file", transformation_file
        experiment = Experiment()
        experiment.calibration = calibration_set
        experiment.transformation = transformation_file
        experiment.from_pair = from_pair
        experiment.to_pair = to_pair
        experiments.append(experiment)
    return experiments


def runExperiment(experiment):
    points_file_from = "points" + experiment.from_pair.numString() + ".pcd"
    descriptors_file_from = "descriptors" + experiment.from_pair.numString() + ".pcd"
    points_file_to = "points" + experiment.to_pair.numString() + ".pcd"
    descriptors_file_to = "descriptors" + experiment.to_pair.numString() + ".pcd"
    transform_file = experiment.from_pair.numString() + "to" + experiment.to_pair.numString() + "-est.txt"
    cmd = ["rosrun", "stereo_feature_extraction", "extractor"]
    cmd.append("-L")
    cmd.append(experiment.from_pair.left)
    cmd.append("-R")
    cmd.append(experiment.from_pair.right)
    cmd.append("-J")
    cmd.append(experiment.calibration.left)
    cmd.append("-K")
    cmd.append(experiment.calibration.right)
    cmd.append("-C")
    cmd.append(points_file_from)
    cmd.append("-D")
    cmd.append(descriptors_file_from)
    print "Running extractor for 'from' image pair..."
    print " ".join(cmd)
    if subprocess.call(cmd) != 0:
        print "ERROR running extractor!"
        sys.exit(2)
    cmd[4] = experiment.to_pair.left
    cmd[6] = experiment.to_pair.right
    cmd[12] = points_file_to
    cmd[14] = descriptors_file_to
    print "Running extractor for 'to' image pair..."
    print " ".join(cmd)
    if subprocess.call(cmd) != 0:
        print "ERROR running extractor!"
        sys.exit(2)
    print "Running transformation_estimator..."
    cmd = ["rosrun", "object_detection", "transformation_estimator"]
    cmd.append("-P")
    cmd.append(points_file_from)
    cmd.append("-Q")
    cmd.append(points_file_to)
    cmd.append("-F")
    cmd.append(descriptors_file_from)
    cmd.append("-G")
    cmd.append(descriptors_file_to)
    cmd.append("-T")
    cmd.append(transform_file)
    print " ".join(cmd)
    if subprocess.call(cmd) != 0:
        print "ERROR running transformation_estimator!"
        sys.exit(2)
    cmd = ["rosrun", "object_detection", "transformation_error_calculator"]
    cmd.append("-P")
    cmd.append(points_file_from)
    cmd.append("-T")
    cmd.append(experiment.transformation)
    cmd.append("-E")
    cmd.append(transform_file)
    print " ".join(cmd)
    if subprocess.call(cmd) != 0:
        print "ERROR running transformation_error_calculator!"
        sys.exit(2)


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print >>sys.stderr, "Usage: {0} <test data folder> [<config file> = defaults.cfg]".format(argv[0])
        return 1

    data_path = argv[1]

    if not os.path.exists(data_path):
        print >>sys.stderr, "ERROR: Path {0} not found!".format(argv[1])
        return 1

    print "Using test data folder {0}".format(data_path)

    if len(argv) == 3:
        config_file = argv[2]
    else:
        config_file = 'defaults.cfg'

    if not os.path.exists(config_file):
        print >>sys.stderr, "ERROR: Config file {0} does not exist, please specify a config file!".format(config_file)
        return 1

    print "Using config file", config_file

    experiments = collectExperiments(data_path)
    num_experiments = len(experiments)
    print "Found data for {0} experiments".format(num_experiments)
    i = 0
    for experiment in experiments:
        print "***** Running experiment {0} of {1}... *****".format(i, num_experiments)
        runExperiment(experiments[i])
        i = i + 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))


