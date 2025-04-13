# SEE: See Everything Every Time 🌟👀
Welcome to the **SEE** project repository! We introduce a new framework for enhancing images across a broad light range using event-based cameras. Our approach adjusts brightness dynamically to enhance image quality in both low and high light conditions. This repository contains the data and code for the **SEE-600K dataset** and **SEE-Net framework**.

## 🎯 Overview
In this project, we explore how to use event-based cameras, with their high dynamic range (HDR), to handle images under various lighting conditions. We propose the **SEE-600K dataset**—a large-scale dataset capturing images under a range of lighting conditions—and a novel framework, **SEE-Net**, that can adjust brightness smoothly and effectively using events.



---

## 📚 Paper Summary
In our research paper, we:
- Collect the **SEE-600K dataset** consisting of 610,126 images and corresponding events under 202 scenarios with a range of lighting conditions.
- Introduce the **SEE-Net framework**, which effectively adjusts image brightness using event-based data.
- Demonstrate that our framework performs well across a broad range of lighting, from very low to very high light levels.
- Show the flexibility of our method through pixel-level brightness adjustments, which can be useful for various post-processing applications.

---

## 📂 Data Availability
The datasets supporting the results of this article are publicly available:
- **SEE-600K Dataset**: [SEE-600K OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/ylu066_connect_hkust-gz_edu_cn/EkNi59p2uHJFjxyeQraiVhgBSs1GnxK4DyCUP-uZhEspCA?e=ZpwOvY)
- **SDE Dataset**: [SDE Dataset GitHub](https://github.com/EthanLiang99/EvLight)

---

## 🛠️ Code & Framework

### 🎥 SEE-Net Framework

The **SEE-Net** framework is designed to adjust the brightness of images under a variety of lighting conditions. It utilizes event-based data to enhance low-light and high-light images and improve their overall quality.

Key Features:
- **Brightness Adjustment** 🌈: Our framework allows for pixel-level brightness adjustment based on the events captured by the camera.
- **Compact & Efficient** ⚡: SEE-Net is designed with efficiency in mind, with only 1.9 million parameters, making it suitable for real-time applications.

![](./images/WX20250227-175014@2x.png)

### 🖥️ Installation

To get started with this repository, clone it using the following command:

```bash
git clone https://github.com/yunfanLu/SEE.git
```

Next, install the required dependencies:

```bash
pip install -r requirements.txt
```

You can then use the code to process images using our **SEE-Net** framework and adjust their brightness using the **SEE-600K** dataset.

---

## 💬 Usage

1. **Download the SEE-600K dataset** using the link provided above.
2. **Preprocess the data** using the provided Python scripts to prepare the dataset for training.
3. **Train the SEE-Net model** to adjust brightness using the provided scripts and experiment with different lighting conditions.
4. **Adjust the brightness** of any image by setting the brightness prompt (B) to control the exposure level.

---

## 🖼️ Example Outputs

Here are some example outputs generated using our framework:

- **Low-Light Enhancement** 🌙: Images processed under very low-light conditions.
- **High-Light Recovery** 🌞: Images processed under overexposed or bright light conditions.


## IMU-Based Registration-Tool

I released the registration tool separately to this repository - [Registration-Tool](https://github.com/yunfanLu/IMU-Registration-Tool).

---

## 📄 Citation

If you use the SEE-600K dataset or SEE-Net in your research, please cite our paper:

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

---

## 🐱‍🏍 Contributions

We welcome contributions to this project! If you have any ideas or improvements, feel free to fork this repository and create a pull request. 💪

- Bug fixes
- New features
- Improvements to documentation

---

## 📞 Contact

If you have any questions or need assistance, feel free to reach out to us!

- **Yunfan Lu**: ylu066@connect.hkust-gz.edu.cn
- **GitHub**: [yunfanLu](https://github.com/yunfanLu)

---

✨ We hope you find the SEE project helpful and look forward to your contributions! 😄
