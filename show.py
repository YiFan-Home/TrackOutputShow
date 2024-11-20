import argparse
import colorsys
import hashlib
import os
import random

import cv2
import numpy as np


video_seq = {
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test':[
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    }
}

MOT17_val_begin = {
    'MOT17-02-FRCNN': 304,
    'MOT17-04-FRCNN': 529,
    'MOT17-05-FRCNN': 422,
    'MOT17-09-FRCNN': 266,
    'MOT17-10-FRCNN': 331,
    'MOT17-11-FRCNN': 454,
    'MOT17-13-FRCNN': 379
}

# 生成颜色
def generate_colors(N):
    # 使用HSV模型生成N个均匀分布的颜色
    colors = []
    for i in range(N):
        # 计算色相值，均匀分布在0到1之间
        hue = i / N
        # 设置饱和度和亮度为固定值（例如 0.8 和 0.8）
        rgb_float = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        # 将RGB从浮动值转换到0-255区间
        rgb = tuple(int(c * 255) for c in rgb_float)
        colors.append(rgb)
    return colors


def generate_video_from_images(image_folder, output_video_file, frame_rate=30):
    # 获取文件夹中的所有JPG格式的图片文件，按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    # images.sort()  # 确保图片按顺序排列
    # 确保文件夹中有图片
    if len(images) == 0:
        print("文件夹中没有JPG图片！")
        return
    # 获取第一张图片的尺寸，作为视频的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    size = (width, height)
    # 定义视频写入器，使用'mp4v'编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v编码器，适用于.mp4文件
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, size)
    # 逐张图片读取并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)
        if img is None:
            continue
        out.write(img)  # 将图片写入视频
    # 释放视频写入器
    out.release()


def _add_box_id_2img(img_path, box_data, id, save_path, id_to_color, ids_last_frame):
    img = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'ids:%d'%ids_last_frame, (2,img.shape[0]-2), font, 1, (0,0,255), 2, cv2.LINE_AA)
    for i, box in zip(id, box_data):
        # 获取目标ID对应的颜色
        color = id_to_color[i-1]
        # 绘制边界框
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)  # 使用对应的颜色
        # 在边界框右上角写上目标ID
        # font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), (box[2], box[1]), font, 1, color, 2, cv2.LINE_AA)
    cv2.imwrite(save_path, img)
    return max(ids_last_frame, np.max(id))

def show_track_output(img_path, video_seq, tracker_path, save_path, GV=False):
    for seq in video_seq:
        print('开始处理：%s'%seq)
        video_path = img_path + seq + '/img1/'
        video_save_path = save_path + seq + '/img/'
        video_track_path = tracker_path + seq + '.txt'
        # id_to_color = {}

        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)

        track_data = np.loadtxt(video_track_path, delimiter=',')
        min_frame = int(np.min(track_data[:,0]))
        max_frame = int(np.max(track_data[:,0]))
        num_id = int(np.max(track_data[:,1]))
        id_to_color = generate_colors(num_id)
        random.shuffle(id_to_color)

        # min_frame += MOT17_val_begin[seq]-3
        # max_frame += MOT17_val_begin[seq]-3
        ids_last_frame = 0
        for frame in range(min_frame, max_frame+1):
            # frame_img = video_path + '%06d.jpg' % frame
            frame_img = video_path + '%06d.jpg'%(frame + MOT17_val_begin[seq] - 3)
            frame_track_data = track_data[track_data[:, 0] == frame, 1:6]
            frame_track_data[:,3:5] += frame_track_data[:,1:3]
            frame_track_data = frame_track_data.astype(np.int32)
            # frame_img_save = video_save_path + '%06d.jpg'%frame
            frame_img_save = video_save_path + '%06d.jpg' % (frame + MOT17_val_begin[seq] - 3)
            ids_last_frame = _add_box_id_2img(frame_img, frame_track_data[:,1:5], frame_track_data[:,0], frame_img_save, id_to_color, ids_last_frame)
        # 生成视频
        if GV:
            print("开始生成视频")
            video_output_path = save_path + seq + '/%s.mp4'%seq
            generate_video_from_images(video_save_path, video_output_path)


def get_args():
    # 用来装载参数的容器
    parser = argparse.ArgumentParser(description='Visualize tracking results')
    # 给这个解析对象添加命令行参数
    parser.add_argument('-dataset', type=str, default='MOT17', help='dataset: MOT17/MOT20')
    parser.add_argument('-mod', type=str, default='val', help='train/val/test')
    parser.add_argument('-input', type=str, default='track', help='gt/det/track')
    parser.add_argument('-tracker', type=str, default=None, help='Tracker name')
    parser.add_argument('--Video', action='store_true', help='Generate videos')
    args = parser.parse_args()  # 获取所有参数

    args.img_path = 'datasets/' + args.dataset
    if args.mod == 'test':
        args.img_path += '/test/'
    else:
        args.img_path += '/train/'

    # args.save_path = 'output_imgs/' + args.dataset + '-' + args.mod + '/'
    args.video_seq = video_seq[args.dataset][args.mod]

    return args

def main():
    args = get_args()
    if args.input == 'gt':
        print("可视化gt")
    elif args.input == 'det':
        print("可视化det")
    else:
        tracker_path = 'outputs/' + args.dataset + '-' + args.mod + '/' + args.tracker + '/'
        args.save_path = 'output_imgs/' + args.dataset + '-' + args.mod + '/' + args.tracker + '/'
        show_track_output(args.img_path, args.video_seq, tracker_path, args.save_path, args.Video)


if __name__ == '__main__':
    main()
















"""
import cv2
import hashlib

# 读取txt文件的路径
txt_file_path = 'data.txt'

# 图像文件路径的格式 (假设图像是以帧数命名，如frame_1.jpg)
image_file_path_format = 'frame_{}.jpg'

# 用于存储每个目标ID的颜色映射
id_to_color = {}

# 函数：通过ID生成一个唯一的颜色
def generate_color_for_id(target_id):
    # 使用SHA-256生成哈希值
    hash_object = hashlib.sha256(str(target_id).encode())
    hex_dig = hash_object.hexdigest()
    
    # 取哈希值的前6个字符来生成RGB
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    
    return (b, g, r)  # 返回(B, G, R)

# 读取txt文件
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# 遍历每一行数据
for line in lines:
    # 每行数据以逗号分隔
    data = line.strip().split(',')
    
    # 提取数据
    frame_number = int(data[0])  # 视频帧数
    target_id = int(data[1])     # 目标ID（这里转为整数）
    x = int(data[2])             # 边界框左上角横坐标
    y = int(data[3])             # 边界框左上角纵坐标
    w = int(data[4])             # 边界框宽度
    h = int(data[5])             # 边界框高度
    
    # 检查目标ID是否已分配颜色
    if target_id not in id_to_color:
        # 如果没有为该ID分配颜色，则生成一个唯一颜色
        id_to_color[target_id] = generate_color_for_id(target_id)
    
    # 获取目标ID对应的颜色
    color = id_to_color[target_id]
    
    # 加载对应的视频帧图像
    image = cv2.imread(image_file_path_format.format(frame_number))
    
    if image is None:
        print(f"Error: Image for frame {frame_number} not found.")
        continue
    
    # 绘制边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # 使用对应的颜色
    
    # 在边界框右上角写上目标ID
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(target_id), (x + w, y), font, 1, color, 2, cv2.LINE_AA)
    
    # 显示图像（可以选择保存图像）
    cv2.imshow(f'Frame {frame_number}', image)
    
    # 等待键盘事件，按'esc'键退出
    if cv2.waitKey(0) & 0xFF == 27:
        break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
"""