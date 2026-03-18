import cv2
import numpy as np
from matplotlib import pyplot as plt

img_blk = cv2.imread('2.png')
(B, G, R) = cv2.split(img_blk)
cen_B = np.mean(B)
std_B = np.std(B)
cen_G = np.mean(G)
std_G = np.std(G)
cen_R = np.mean(R)
std_R = np.std(R)

img = cv2.imread("images/1.jpg")
rows = img.shape[0]
cols = img.shape[1]

mask = np.zeros(img.shape)
for x in range(rows):
    for y in range(cols):
        b, g, r = img[x, y]
        flag_b = (b <= cen_B+1.25*std_B) and (b >= cen_B-1.25*std_B)
        flag_g = (g <= cen_G + 1.25 * std_G) and (g >= cen_G - 1.25 * std_G)
        flag_r = (r <= cen_R + 1.25 * std_R) and (r >= cen_R - 1.25 * std_R)
        if flag_b and flag_g and flag_r:
            mask[x, y] = 1

# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('original')
#
# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(img_blk, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('sample')

# plt.subplot(1, 3, 3)

plt.imshow(mask, cmap='gray')
plt.axis('off')
# plt.title('segment')
plt.savefig("666.png",format='png', dpi=100)
plt.show()

