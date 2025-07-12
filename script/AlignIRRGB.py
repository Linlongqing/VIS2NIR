import os
import cv2
import numpy as np
import random 
from skimage import transform as trans


def get_landmark(path):
    landmark_path = path.replace('.png', '.txt')
    if not os.path.exists(landmark_path):
        return None, None

    lines = open(landmark_path).readlines()
    landmark_info = lines[1].strip().split(' ')
    points = []
    for i in range(len(landmark_info) // 2):
        x = float(landmark_info[2 * i])
        y = float(landmark_info[2 * i + 1])
        points.append([x, y])
    points = np.array(points)
    blur = float(lines[0].split(' ')[1])
    return points, blur

def draw_landmark(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (255), thickness=-1)


def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def get_angle_and_center(keypoints):
    # 假设第37和46个点是左右眼的中心点
    left_eye = (keypoints[36] + keypoints[39]) // 2
    right_eye = (keypoints[42] + keypoints[45]) // 2

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    return angle, center

def crop_face2(image, keypoints):
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    left_eye = (keypoints[20] + keypoints[54]) // 2
    right_eye = (keypoints[58] + keypoints[69]) // 2
    mouth_left = keypoints[36]
    mouth_right = keypoints[45]
    nose = keypoints[63]
    landmark = np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32)
    # print(landmark)

    dst[:,0] += 8.0
    dst = dst * (256.0 / 112.0)
    src = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(image, M, (256,256), borderValue=0.0)    
    return img

def align_face_by_all_points(src_img, dst_img, src_points, dst_points):
    """
    根据源图像和目标图像的关键点进行对齐。
    :param src_img: 源图像
    :param dst_img: 目标图像（用于获取目标关键点）
    :param src_points: 源图像的关键点坐标
    :param dst_points: 目标图像的关键点坐标
    :return: 对齐后的图像
    """

    output_size = (src_img.shape[1], src_img.shape[0])
    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    if M is None:
        print("无法计算仿射变换矩阵")
        return None
    # 应用仿射变换
    aligned_img = cv2.warpAffine(src_img, M, output_size, flags=cv2.INTER_CUBIC)
    src_transformed = cv2.transform(src_points.reshape(-1, 1, 2), M).reshape(-1, 2)
    errors = np.linalg.norm(src_transformed - dst_points, axis=1)
    mean_error = np.mean(errors)
    return aligned_img, mean_error

def concatenate_images_horizontally(img1, img2):
    # 获取两张图片的高度和宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_img = np.zeros((max(h1, h2), w1 + w2, 1), dtype=np.uint8)
    min_height = min(h1, h2)
    new_img[:h1, :w1, :] = img1.reshape(256, 256, 1)
    new_img[:h2, w1:, :] = img2.reshape(256, 256, 1)
    return new_img

root = 'your_path/face_rgb_labeled'
ref_root = 'your_path/face_ir_labeled'

dst_root = 'your_path/pix2pixHD/datasets/align_faces'

dst_train_img_dir = os.path.join(dst_root, 'train_A')
dst_train_label_dir = os.path.join(dst_root, 'train_B')
dst_test_img_dir = os.path.join(dst_root, 'test_A')
dst_test_label_dir = os.path.join(dst_root, 'test_B')
if not os.path.exists(dst_train_img_dir):
    os.mkdir(dst_train_img_dir)
if not os.path.exists(dst_train_label_dir):
    os.mkdir(dst_train_label_dir)
if not os.path.exists(dst_test_img_dir):
    os.mkdir(dst_test_img_dir)
if not os.path.exists(dst_test_label_dir):
    os.mkdir(dst_test_label_dir)

cnt = 0
users = os.listdir(root)
for user in users:
    user_path = os.path.join(root, user)
    files = os.listdir(user_path)
    for file in files:
        img_path = os.path.join(user_path, file)
        if file.endswith('.txt'):
            continue
        ref_img_name = file.replace('@rgb', '@ir')
        ref_img_path = os.path.join(ref_root, user, ref_img_name)
        rgb_image = cv2.imread(img_path, 0)
        ir_image = cv2.imread(ref_img_path, 0)
        rgb_points, rgb_blur = get_landmark(img_path)
        ir_points, ir_blur = get_landmark(ref_img_path)

        if rgb_points is None or ir_points is None:
            continue
        if rgb_blur > 0.5 or ir_blur > 0.5:
            continue

        aligned_img,mean_error = align_face_by_all_points(rgb_image, ir_image, rgb_points, ir_points)
        if mean_error > 1.5:
            continue
        cropped_face_ir = crop_face2(ir_image, ir_points)
        cropped_face_rgb = crop_face2(aligned_img, ir_points)        

        if random.randint(0, 10) < 1:
            cv2.imwrite(os.path.join(dst_test_img_dir, file), cropped_face_rgb)
            cv2.imwrite(os.path.join(dst_test_label_dir, file), cropped_face_ir)
        else:
            cv2.imwrite(os.path.join(dst_train_img_dir, file), cropped_face_rgb)
            cv2.imwrite(os.path.join(dst_train_label_dir, file), cropped_face_ir)
        cnt += 1
        if cnt % 100 == 0:
            print('proccess:', cnt)
        # if cnt > 400:
        #     break
        # dst_image_path = 
        # merged_image = cv2.merge((cropped_face_ir, cropped_face_rbg, cropped_face_ir))
        # cv2.imwrite('merge.png', merged_image)
    #     break
    # break
