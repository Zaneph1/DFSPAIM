import matplotlib.pyplot as plt

# create grid
data = [[4,4,4,4],
        [4,4,4,4],
        [4,4,4,4],
        ]

import matplotlib.pyplot as plt
import numpy as np

# example data
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]
z = [[0,0,3,0],
    [0,0,3,0],
    [0,0,0,0],
    [1,1,0,0]
        ]

# create a 2D grid
X, Y = np.meshgrid(x, y)

# draw grid distribution
plt.pcolormesh(X, Y, z, shading='auto')

# add colorbar
plt.colorbar()

# set axis labels
plt.xlabel('X')
plt.ylabel('Y')

# show plot
plt.show()
# import matplotlib.pyplot as plt
#
# # create grid
# grid = [[4,4,4,4],
#     [4,4,4,4],
#     [4,4,4,4],
#     [4,4,4,4]
#         ]
#
# # fill colors
# grid[2:5, 3:7] = 1
# grid[6:9, 1:4] = 2
#
# #
# plt.imshow(grid, cmap='Paired', interpolation='nearest')
# plt.grid(True, which='both', color='black', linewidth=1)
# plt.xticks(np.arange(0, 4, 1))
# plt.yticks(np.arange(0, 4, 1))
# plt.show()
