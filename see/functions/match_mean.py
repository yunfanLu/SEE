import torch
import torch.nn.functional as F


def search_gamma_to_match_mean(image1, image2, max_iter=10):
    # downsample the size of the image to speed up the process
    image1 = F.avg_pool2d(image1, kernel_size=10, stride=10)
    image2 = F.avg_pool2d(image2, kernel_size=10, stride=10)

    target_mean = torch.mean(image2)
    lower_gamma, upper_gamma = 0.1, 10
    best_gamma = 1.0

    for i in range(max_iter):
        gamma = (lower_gamma + upper_gamma) / 2
        adjusted_image1 = torch.pow(image1, gamma)
        adjusted_mean = torch.mean(adjusted_image1)
        if adjusted_mean < target_mean:
            lower_gamma = gamma
        else:
            upper_gamma = gamma
        best_gamma = gamma
    return best_gamma
