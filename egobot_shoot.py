import argparse
import os
import time
import cv2


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def main(args):
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    # Get lastest frame
    sorted_frames = sorted(os.listdir(args.path), key=lambda x: int(x.split('.')[0].split('-')[-1]))
    if len(sorted_frames) >= 1:
        img_pointer = int(sorted_frames[-1].split('.')[0].split('-')[-1]) + 1
    else:
        img_pointer = 0

    counter = 0
    img_counter = 0
    interval = int(args.interval / 0.001)
    total_count = round(args.shot_length / args.interval)
    print("Ready to shoot: total {} images with {}s interval!\n".format(total_count, args.interval))
    #print(cv2.CAP_PROP_POS_FRAMES, cv2.CAP_PROP_POS_MSEC)
    
    ret = True
    
    if cam.isOpened():
        try:
            #window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while ret:
                ret, frame = cam.read()
                counter += 1
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
                
                if counter >= (img_counter * 30 * args.interval):
                    img_name = "frame-{}.png".format(img_counter + img_pointer)
                    cv2.imwrite(os.path.join(args.path, img_name), frame)
                    print("{} written!".format(img_name))
                    # Move FPS forward
                    img_counter += 1
                    #cam.set(cv2.CAP_PROP_POS_FRAMES, (img_counter * 30 * args.interval))
                
                
                if img_counter >= total_count:
                    break
                
        finally:
            cam.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser(description='Specify image saving folder and inference engine')
    
    my_parser.add_argument('--path',
                       type=str,
                       required=False,
                       default="/home/nvidia/Pictures/EgoShot/",
                       help='the path to image saving folder')
    my_parser.add_argument('--shot_length',
                       type=int,
                       required=False,
                       default=300,
                       help='total period of shooting (s)')
    my_parser.add_argument('--interval',
                       type=int,
                       required=False,
                       default=30,
                       help='shooting interval (s)')
    args = my_parser.parse_args()
    
    main(args)

