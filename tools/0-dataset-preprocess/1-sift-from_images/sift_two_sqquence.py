import cv2
import numpy as np


def extract_features_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    distances = [match.distance for match in matches]
    return np.mean(distances), matches


def calculate_rgb_frame_differences(rgb_images1, rgb_images2):
    differences = []
    all_matches = []
    for img1, img2 in zip(rgb_images1, rgb_images2):
        keypoints1, descriptors1 = extract_features_from_image(img1)
        keypoints2, descriptors2 = extract_features_from_image(img2)

        if descriptors1 is not None and descriptors2 is not None:
            mean_distance, matches = match_features(descriptors1, descriptors2)
            differences.append(mean_distance)
            all_matches.append(matches)
        else:
            differences.append(float("inf"))
            all_matches.append([])

    return differences, all_matches


def draw_matches(img1, keypoints1, img2, keypoints2, matches, output_path):
    matched_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, matched_img)


def main():
    # 假设rgb_images1和rgb_images2是两个时间序列的RGB图像列表
    # 例如：
    rgb_images1 = [cv2.imread(f"rgb_image_path_1_{i}.png") for i in range(number_of_images)]
    rgb_images2 = [cv2.imread(f"rgb_image_path_2_{i}.png") for i in range(number_of_images)]

    # 计算每对帧之间的特征点匹配偏差
    differences, all_matches = calculate_rgb_frame_differences(rgb_images1, rgb_images2)

    # 输出偏差
    for i, diff in enumerate(differences):
        print(f"Frame {i}: Feature matching mean distance = {diff}")

        # 保存匹配结果图像
        if all_matches[i]:
            keypoints1, descriptors1 = extract_features_from_image(rgb_images1[i])
            keypoints2, descriptors2 = extract_features_from_image(rgb_images2[i])
            draw_matches(
                rgb_images1[i], keypoints1, rgb_images2[i], keypoints2, all_matches[i], f"matched_frame_{i}.png"
            )


if __name__ == "__main__":
    main()
