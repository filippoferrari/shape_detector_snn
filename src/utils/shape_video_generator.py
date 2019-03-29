import argparse
import cv2
import numpy as np
import random

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

    if 'random' in args.shape:
        tot_frames = 0
        while True:
            frames = random.randint(10, 20)

            if tot_frames > 5*FPS:
                break

            if 'square' in args.shape and 'diamond' not in args.shape:
                paint_x = random.randint(0, width-radius-1)
                paint_y = random.randint(0, width-radius-1)
                for i in range(frames):            
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.rectangle(frame, (paint_x, paint_y), (paint_x+radius, paint_y+radius), colour, 1) 
                    video.write(frame)
            elif 'diamond' in args.shape and 'square' not in args.shape:
                r = radius//2 + 1
                paint_x = random.randint(radius, width-radius-1)
                paint_y = random.randint(radius, width-radius-1)
                c = paint_x #+ radius
                pts = np.array([[c,paint_y+r],[c+r, paint_y],[c,paint_y-r], [c-r,paint_y]], np.int32)
                pts = pts.reshape((-1,1,2))
                for i in range(frames):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)      
                    cv2.polylines(frame,[pts],True,colour)
                    video.write(frame)
            elif 'square' in args.shape and 'diamond' in args.shape:
                # Diamond
                r = radius//2 + 1
                paint_x = random.randint(radius, width-radius-1)
                paint_y = random.randint(radius, width-radius-1)
                c = paint_x #+ radius
                pts = np.array([[c,paint_y+r],[c+r, paint_y],[c,paint_y-r], [c-r,paint_y]], np.int32)
                pts = pts.reshape((-1,1,2))

                # Square
                while True:
                    paint_x_s = random.randint(0, width-radius-1)
                    paint_y_s = random.randint(0, width-radius-1)
                    if len(list(set(range(paint_x_s,paint_x_s+radius)) & set(range(c-r,c+r)))) == 0:
                        break
                    if len(list(set(range(paint_y_s,paint_y_s+radius)) & set(range(paint_y-r,paint_y+r)))) == 0:
                        break
            
                for i in range(frames):            
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    # Square
                    cv2.rectangle(frame, (paint_x_s+1, paint_y_s+1), (paint_x_s+radius-1, paint_y_s+radius-1), colour, 1) 
                    # Diamond
                    cv2.polylines(frame,[pts],True,colour)
                    video.write(frame)

                pass

            tot_frames += frames

    else:
        for _ in range(0,3):
            for paint_x in range(-radius, width+radius):
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                if args.shape == 'circle':
                    cv2.circle(frame, (paint_x, paint_h), radius, colour, 1)
                elif args.shape == 'square_lr':
                    cv2.rectangle(frame, (paint_x, paint_h-radius//2), (paint_x+radius-1, paint_h+radius//2), colour, 1) 
                elif args.shape == 'square_tb':
                    cv2.rectangle(frame, (paint_h-radius//2, paint_x), (paint_h+radius//2 , paint_x+radius-1), colour, 1) 
                elif args.shape == 'square_diag':
                    cv2.rectangle(frame, (paint_x-radius//2, paint_x), (paint_x+radius//2 , paint_x+radius-1), colour, 1) 
                elif args.shape == 'vertical':
                    cv2.rectangle(frame, (paint_x, paint_h-radius//2), (paint_x+bar_width, paint_h+radius//2), colour, -1) 
                elif args.shape == 'vertical_big':
                    cv2.rectangle(frame, (paint_x, 0), (paint_x+bar_width, width), colour, -1) 
                elif args.shape == 'horizontal_big':
                    cv2.rectangle(frame, (0, paint_x), (width, paint_x), colour, -1) 
                elif args.shape == 'diamond_lr':
                    c = paint_x + radius
                    r = radius//2 + 1
                    pts = np.array([[c,paint_h+r],[c+r, paint_h],[c,paint_h-r], [c-r,paint_h]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(frame,[pts],True,colour)
                elif args.shape == 'diamond_tb':
                    c = paint_x + radius
                    r = radius//2 + 1
                    pts = np.array([[paint_h+r,c,],[paint_h,c+r],[paint_h-r,c], [paint_h,c-r]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(frame,[pts],True,colour)
                elif args.shape == 'triangle_lr':
                    c = paint_x + radius
                    r = radius//2
                    pts = np.array([[c+r, paint_h],[c-r, paint_h],[c,paint_h-r],], np.int32)
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
