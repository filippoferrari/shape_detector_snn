import argparse
import cv2
import numpy as np


def main(args):
    width = 32
    height = 32
    FPS = 24
    seconds = 10
    radius = 5
    bar_width = 1
    paint_h = int(height/2)

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./' + args.shape.lower() +'.avi', fourcc, float(FPS), (width, height))

    for _ in range(0,3):
        for paint_x in range(-radius, width+radius):
            frame = np.ones((height, width, 3), dtype=np.uint8)*255

            if args.shape == 'circle':
                cv2.circle(frame, (paint_x, paint_h), radius, (0, 0, 0), 1)
            elif args.shape == 'square':
                cv2.rectangle(frame, (paint_x, paint_h-radius//2), (paint_x+radius, paint_h+radius//2), (0,0,0), -1) 
            elif args.shape == 'vertical':
                cv2.rectangle(frame, (paint_x, paint_h-radius//2), (paint_x+bar_width, paint_h+radius//2), (0,0,0), -1) 

            video.write(frame)

    video.release()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--shape', default='circle', required=True, type=str, help='Shape in [\'circle\', \'square\']')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
