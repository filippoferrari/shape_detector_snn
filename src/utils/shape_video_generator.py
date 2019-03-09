import argparse
import cv2
import numpy as np


def main(args):
    width = 32
    height = 32
    FPS = 24
    seconds = 10
    radius = args.edge
    bar_width = 1
    paint_h = int(height/2)
    colour = (255, 255, 255)

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./' + args.shape.lower() + '_' + str(radius) + 'x' + str(radius) +'.avi', fourcc, float(FPS), (width, height))

    for _ in range(0,3):
        for paint_x in range(-radius, width+radius):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if args.shape == 'circle':
                cv2.circle(frame, (paint_x, paint_h), radius, colour, 1)
            elif args.shape == 'square_lr':
                cv2.rectangle(frame, (paint_x, paint_h-radius//2), (paint_x+radius-1, paint_h+radius//2), colour, 1) 
            elif args.shape == 'square_tb':
                cv2.rectangle(frame, (paint_h-radius//2, paint_x), (paint_h+radius//2 , paint_x+radius-1), colour, 1) 
            elif args.shape == 'vertical':
                cv2.rectangle(frame, (paint_x, paint_h-radius//2), (paint_x+bar_width, paint_h+radius//2), colour, -1) 
            elif args.shape == 'diamond_lr':
                c = paint_x + radius
                r = radius//2
                pts = np.array([[c,paint_h+r],[c+r, paint_h],[c,paint_h-r], [c-r,paint_h]], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame,[pts],True,colour)
            elif args.shape == 'diamond_tb':
                c = paint_x + radius
                r = radius//2
                pts = np.array([[paint_h+r,c,],[paint_h,c+r],[paint_h-r,c], [paint_h,c-r]], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame,[pts],True,colour)


            video.write(frame)

    video.release()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--shape', default='circle', required=True, type=str, help='Shape in [\'circle\', \'square\']')
    parser.add_argument('-e', '--edge', default=5, required=False, type=int, help='Length of the edge')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
