# Box detection challenge,
# Developed by: Rayhra Pozo
import numpy as np
import cv2
import json
import argparse
import os
import sys

# Parser for input arguments
def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', type=str, default='./form.jpg',
                      help='Local file path of the input document in which to detect the boxes')
  parser.add_argument('--output_dir', type=str, default='./',
                      help='Local directory where to save output documents')
  return parser


#Intersection over union function to evaluate ovelapping bounding boxes
def iou_fun(box1, box2):
    x_left = max(box1['x1'], box2['x1'])
    y_top = max(box1['y1'], box2['y1'])
    x_right = min(box1['x2'], box2['x2'])
    y_bottom = min(box1['y2'], box2['y2'])

    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    iou = intersection / float(box1['area'] + box2['area'] - intersection)
    assert iou >= 0.0 and iou <= 1.0
    return iou


#Function to write JSON file
def writeJSON(path, filename, data):
    filepath = './' + path + filename + '.json'
    with open(filepath, 'w') as f:
        json.dump(data, f)


#Function to read image, apply filters, bounding boxes generation
def find_boxes(input_file):
    img = cv2.imread(input_file)
    cannyedge = cv2.Canny(img,100,100)
    contours = cv2.findContours(cannyedge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    idx =0
    boxes = []
    for cnt in contours[1]:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area>=1000:
            box = {'x1':x, 'x2':x+w, 'y1':y, 'y2':y+h, 'area':area}
            boxes.append(box)
    return boxes


#Function to filter overlapping boxes
def filter_boxes(boxes):
    boxes.sort(key=lambda x:x['area'])
    filtered_box_set = []
    for box in boxes:
        if not filtered_box_set:
            filtered_box_set.append(box)
        else:
            for box2 in filtered_box_set:
                iou = iou_fun(box, box2)
                if iou>=0.5:
                    break
            else:
                filtered_box_set.append(box)
    return filtered_box_set


#Function to draw bounding boxes, generate JSON file and image output
def generate_output(input_file, output_dir, filtered_box_set):
    filename = os.path.basename(input_file)[:-4]
    img = cv2.imread(input_file)
    idx = 0
    data = {'boxes':[]}
    for box in filtered_box_set:
        cv2.rectangle(img, (box['x1'], box['y1']),(box['x2'], box['y2']),
                     (238,130,238), 2)
        cv2.putText(img, str(idx), ((box['x1'] + box['x2'])//2,
                   (box['y1'] + box['y2'])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (238,130,238),2)
        data['boxes'].append({'box_{}'.format(idx):box})
        idx += 1
    cv2.imwrite(output_dir + filename + '.jpg', img)
    writeJSON(output_dir, filename, data)


if __name__=='__main__':
    args = create_parser().parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    if not os.path.exists(input_file): sys.exit('Input file does not exist')
    if not os.path.isdir(output_dir): sys.exit('Output directory does not exist')
    boxes = find_boxes(input_file)
    filtered_box_set = filter_boxes(boxes)
    generate_output(input_file, output_dir, filtered_box_set)
    print('Done.')
