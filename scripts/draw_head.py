# %%
import pyvista as pv
import numpy as np

data = np.load("../output.npy")
# %%
print(data.shape)
# 32 32 97 97
i1 = data[:,0,:,:]

# %%
# 转换为 PyVista 格式
grid = pv.ImageData()
grid.dimensions = i1.shape
grid.point_data["values"] = i1.flatten(order="F")

# 渲染体数据
plotter = pv.Plotter()
plotter.add_volume(grid, cmap="viridis")
plotter.show()
# %%
