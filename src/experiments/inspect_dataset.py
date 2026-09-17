import cv2
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 4, figsize=(12, 25))

for num in range(4):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for group in range(3):
        for i in range(4):
            index = num * 16 + group * 4 + i
            path = f"../dataset/ukbench/ukbench{index:05d}.jpg"
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[group, i].imshow(image)
            axes[group, i].set_title(f"Group {group}, Image {i}")
            axes[group, i].axis("off")
    plt.tight_layout()
    plt.show()