import jetson.inference
import jetson.utils

import argparse

parser = argparse.ArgumentParser(description="Draw a dot where the person in an iamge is pointing")
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("output", type=str, help="filename of the image to output")
opt = parser.parse_args()

img = jetson.utils.loadImage(opt.filename)
output = jetson.utils.videoOutput(opt.output)
net = jetson.inference.poseNet()
poses = net.Process(img)

#l_x = 0
#l_y = 0
#r_x = 0
#r_y = 0

for pose in poses:

    left_wrist_idx = pose.FindKeypoint(net.FindKeypointID('left_wrist'))
    left_shoulder_idx = pose.FindKeypoint(net.FindKeypointID('left_shoulder'))

    right_wrist_idx = pose.FindKeypoint(net.FindKeypointID('right_wrist'))
    right_shoulder_idx = pose.FindKeypoint(net.FindKeypointID('right_shoulder'))

    if left_wrist_idx < 0 or left_shoulder_idx < 0 or right_wrist_idx < 0 or right_shoulder_idx < 0:
        continue

    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]
    right_wrist = pose.Keypoints[right_wrist_idx]
    right_shoulder = pose.Keypoints[right_shoulder_idx]

    l_point_x = left_wrist.x - left_shoulder.x
    l_point_y = left_wrist.y - left_shoulder.y
    r_point_x = right_wrist.x - right_shoulder.x
    r_point_y = right_wrist.y - right_shoulder.y

    l_x = left_wrist.x + l_point_x
    l_y = left_wrist.y + l_point_y
    r_x = right_wrist.x + r_point_x
    r_y = right_wrist.y + r_point_y

    jetson.utils.cudaDrawCircle(img, (l_x, l_y), 10, (255, 0, 0, 200))
    jetson.utils.cudaDrawCircle(img, (r_x, r_y), 10, (255, 0, 0, 200))

    output.Render(img)

#    print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")

