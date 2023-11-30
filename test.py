
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

scale = 235.5
# Replace 'path/to/your/file.las' with the path to your LAS file
las_file_path = 'C:/Users/harsh/Downloads/PreFiredata.las'

# Define the x, y range
x_range = (633.75+607000, 869.25+607000)  # Example x range
y_range = (702.5+4894000, 938+4894000)   # Example y range

# Read LAS file
las = laspy.read(las_file_path)

# Filter points based on x, y range
selected_points = np.logical_and.reduce([
    (las.x >= x_range[0]),
    (las.x <= x_range[1]),
    (las.y >= y_range[0]),
    (las.y <= y_range[1])
])

# Extract relevant attributes for the selected points
filtered_points = np.array((las.x[selected_points], las.y[selected_points], las.z[selected_points], las.classification[selected_points])).transpose()

# Create a DataFrame from the filtered points
df = pd.DataFrame(filtered_points, columns=['x', 'y', 'z', 'classification'])

# Filter ground points (classification code 2)
ground_points = df[las.classification[selected_points] == 2]



# Create a Digital Elevation Model (DEM)
dem_resolution = 1.0  # Set the resolution of the DEM
# Combine ground and vegetation points for coordinate grid creation
all_points = df

# Calculate extent of the data
x_min = min(all_points['x'])
x_max = max(all_points['x'])
y_min = min(all_points['y'])
y_max = max(all_points['y'])



# Create a coordinate grid with 250x250 points
x_grid = np.linspace(x_min, x_max, 250)
y_grid = np.linspace(y_min, y_max, 250)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)


# Create DEM using griddata for more accurate interpolation
dem = griddata((ground_points['x'], ground_points['y']), ground_points['z'], (x_mesh, y_mesh), method='linear')

# Apply a median filter to smooth the DEM
dem = median_filter(dem, size=3)

# Create DSM using griddata for more accurate interpolation
dsm = griddata((all_points['x'], all_points['y']), all_points['z'], (x_mesh, y_mesh), method='linear')

# Apply a median filter to smooth the DSM
dsm = median_filter(dsm, size=3)

# Calculate Canopy Height Model (CHM)
chm = dsm - dem


# Filter vegetation points
vegetation_points = df[las.classification[selected_points] == 1]



# Extract x, y, and z coordinates from ground points
ground_x = ground_points['x'].values
ground_y = ground_points['y'].values
ground_z = ground_points['z'].values

num_points_x = int((max(ground_x) - min(ground_x)) / dem_resolution) + 1
num_points_y = int((max(ground_y) - min(ground_y)) / dem_resolution) + 1

# Create a grid for interpolation
grid_x, grid_y = np.meshgrid(np.linspace(min(ground_x), max(ground_x), num_points_x),
                             np.linspace(min(ground_y), max(ground_y), num_points_y))

# Use a different interpolation method (e.g., 'cubic') and handle nan values with fill_value
interpolated_ground_z = griddata((ground_x, ground_y), ground_z, (x_mesh, y_mesh), method='nearest', fill_value=np.nan)




# Extract x, y, and z coordinates from vegetation points
vegetation_x = vegetation_points['x'].values
vegetation_y = vegetation_points['y'].values
vegetation_z = vegetation_points['z'].values

# Interpolate ground heights at vegetation points
interpolated_vegetation_z = griddata((x_mesh.flatten(), y_mesh.flatten()), interpolated_ground_z.flatten(),
                                    (vegetation_x, vegetation_y), method='linear')


# Calculate vegetation heights
vegetation_heights_above_ground = vegetation_z - interpolated_vegetation_z




vegetation_heights_above_ground = np.where(vegetation_heights_above_ground < 0, 0, vegetation_heights_above_ground)





# Calculate the percentiles for vegetation height above ground
percentile_0 = 0.5
percentile_25 = 5
percentile_50 = 10
percentile_75 = 15
percentile_100 = 45


# Create masks for each height percentile range
mask_0_25 = (vegetation_heights_above_ground >= percentile_0) & (vegetation_heights_above_ground < percentile_25)
mask_25_50 = (vegetation_heights_above_ground >= percentile_25) & (vegetation_heights_above_ground < percentile_50)
mask_50_75 = (vegetation_heights_above_ground >= percentile_50) & (vegetation_heights_above_ground < percentile_75)
mask_75_100 = (vegetation_heights_above_ground >= percentile_75) & (vegetation_heights_above_ground <= percentile_100)

# Calculate the grid spacing based on the desired grid size and the minimum and maximum coordinates
desired_grid_size = 250
x_spacing = (max(ground_points['x']) - min(ground_points['x'])) / (desired_grid_size+1)
y_spacing = (max(ground_points['y']) - min(ground_points['y'])) / (desired_grid_size +1)

# Construct the bins using the calculated spacing
x_grid = np.arange(min(ground_points['x']), max(ground_points['x']), x_spacing)
y_grid = np.arange(min(ground_points['y']), max(ground_points['y']), y_spacing)

# Create the histogram using the 250x250 bins
density_0_25, _, _ = np.histogram2d(
    vegetation_points.loc[mask_0_25, 'x'], vegetation_points.loc[mask_0_25, 'y'],
    bins=[x_grid, y_grid],
    density=False
)

density_25_50, _, _ = np.histogram2d(
    vegetation_points.loc[mask_25_50, 'x'], vegetation_points.loc[mask_25_50, 'y'],
    bins=[x_grid, y_grid],
    density=False
)

density_50_75, _, _ = np.histogram2d(
    vegetation_points.loc[mask_50_75, 'x'], vegetation_points.loc[mask_50_75, 'y'],
    bins=[x_grid, y_grid],
    density=False
)
density_75_100, _, _ = np.histogram2d(
    vegetation_points.loc[mask_75_100, 'x'], vegetation_points.loc[mask_75_100, 'y'],
    bins=[x_grid, y_grid],
    density=False
)



total_vegetation_points, _, _ = np.histogram2d(
    vegetation_points['x'], vegetation_points['y'],
    bins=[x_grid, y_grid],
    density=False
)
points = df
total_points, _, _ = np.histogram2d(
    points['x'], points['y'],
    bins=[x_grid, y_grid],
    density=False
)

# Create rasters for each normalized density
normalized_density_overall = np.divide(total_vegetation_points, total_points, out=np.zeros_like(total_points), where=total_points != 0)
normalized_density_0_25 = np.divide(density_0_25, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)
normalized_density_25_50 = np.divide(density_25_50, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)
normalized_density_50_75 = np.divide(density_50_75, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)
normalized_density_75_100 = np.divide(density_75_100, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)



# Display DSM, DEM, and CHM as images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(dem, cmap='terrain',extent=(0, scale, 0, scale))
plt.title('DEM (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')


plt.subplot(1, 2, 2)
plt.imshow(dsm, cmap='viridis',extent=(0, scale, 0, scale))
plt.title('DSM (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('dem1_pre.png')


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(chm, cmap='YlGnBu',extent=(0, scale, 0, scale))
plt.title('Canopy Height (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_overall),k=-1), cmap='viridis',extent=(0, scale, 0, scale))
plt.title('Overall Vegetation Density')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')


plt.savefig('dem2_pre.png')

# Display density rasters
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.rot90(np.flipud(normalized_density_0_25),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (0.5 - 7.5 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_25_50),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (7.5 - 15 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('veg1_pre.png')


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.rot90(np.flipud(normalized_density_50_75),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (15 - 22.5 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_75_100),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (22.5+ m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')


plt.savefig('veg2_pre.png')









# Replace 'path/to/your/file.las' with the path to your LAS file
las_file_path = 'C:/Users/harsh/Downloads/PostFire.las'


# Read LAS file
las = laspy.read(las_file_path)

# Filter points based on x, y range
selected_points = np.logical_and.reduce([
    (las.x >= x_range[0]),
    (las.x <= x_range[1]),
    (las.y >= y_range[0]),
    (las.y <= y_range[1])
])

# Extract relevant attributes for the selected points
filtered_points = np.array((las.x[selected_points], las.y[selected_points], las.z[selected_points], las.classification[selected_points])).transpose()

# Create a DataFrame from the filtered points
df = pd.DataFrame(filtered_points, columns=['x', 'y', 'z', 'classification'])

# Filter ground points (classification code 2)
ground_points = df[las.classification[selected_points] == 2]



# Create a Digital Elevation Model (DEM)
dem_resolution = 1.0  # Set the resolution of the DEM
# Combine ground and vegetation points for coordinate grid creation
all_points = df

# Calculate extent of the data
x_min = min(all_points['x'])
x_max = max(all_points['x'])
y_min = min(all_points['y'])
y_max = max(all_points['y'])



# Create a coordinate grid with 250x250 points
x_grid = np.linspace(x_min, x_max, 250)
y_grid = np.linspace(y_min, y_max, 250)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)


# Create DEM using griddata for more accurate interpolation
dem_post = griddata((ground_points['x'], ground_points['y']), ground_points['z'], (x_mesh, y_mesh), method='linear')

# Apply a median filter to smooth the DEM
dem_post = median_filter(dem_post, size=3)

# Create DSM using griddata for more accurate interpolation
dsm_post = griddata((all_points['x'], all_points['y']), all_points['z'], (x_mesh, y_mesh), method='linear')

# Apply a median filter to smooth the DSM
dsm_post = median_filter(dsm_post, size=3)

# Calculate Canopy Height Model (CHM)
chm_post = dsm_post - dem_post


# Filter vegetation points
vegetation_points = df[las.classification[selected_points] == 1]

# Extract x, y, and z coordinates from ground points
ground_x = ground_points['x'].values
ground_y = ground_points['y'].values
ground_z = ground_points['z'].values

num_points_x = int((max(ground_x) - min(ground_x)) / dem_resolution) + 1
num_points_y = int((max(ground_y) - min(ground_y)) / dem_resolution) + 1

# Create a grid for interpolation
grid_x, grid_y = np.meshgrid(np.linspace(min(ground_x), max(ground_x), num_points_x),
                             np.linspace(min(ground_y), max(ground_y), num_points_y))

# Use a different interpolation method (e.g., 'cubic') and handle nan values with fill_value
interpolated_ground_z = griddata((ground_x, ground_y), ground_z, (x_mesh, y_mesh), method='nearest', fill_value=np.nan)


# Extract x, y, and z coordinates from vegetation points
vegetation_x = vegetation_points['x'].values
vegetation_y = vegetation_points['y'].values
vegetation_z = vegetation_points['z'].values

# Interpolate ground heights at vegetation points
interpolated_vegetation_z = griddata((x_mesh.flatten(), y_mesh.flatten()), interpolated_ground_z.flatten(),
                                    (vegetation_x, vegetation_y), method='linear')


# Calculate vegetation heights
vegetation_heights_above_ground = vegetation_z - interpolated_vegetation_z

vegetation_heights_above_ground = np.where(vegetation_heights_above_ground < 0, 0, vegetation_heights_above_ground)


# Create masks for each height percentile range
mask_0_25 = (vegetation_heights_above_ground >= percentile_0) & (vegetation_heights_above_ground < percentile_25)
mask_25_50 = (vegetation_heights_above_ground >= percentile_25) & (vegetation_heights_above_ground < percentile_50)
mask_50_75 = (vegetation_heights_above_ground >= percentile_50) & (vegetation_heights_above_ground < percentile_75)
mask_75_100 = (vegetation_heights_above_ground >= percentile_75) & (vegetation_heights_above_ground <= percentile_100)

# Calculate the grid spacing based on the desired grid size and the minimum and maximum coordinates
desired_grid_size = 250
x_spacing = (max(ground_points['x']) - min(ground_points['x'])) / (desired_grid_size+1)
y_spacing = (max(ground_points['y']) - min(ground_points['y'])) / (desired_grid_size +1)

# Construct the bins using the calculated spacing
x_grid = np.arange(min(ground_points['x']), max(ground_points['x']), x_spacing)
y_grid = np.arange(min(ground_points['y']), max(ground_points['y']), y_spacing)

# Create the histogram using the 250x250 bins
density_0_25, _, _ = np.histogram2d(
    vegetation_points.loc[mask_0_25, 'x'], vegetation_points.loc[mask_0_25, 'y'],
    bins=[x_grid, y_grid],
    density=False
)

density_25_50, _, _ = np.histogram2d(
    vegetation_points.loc[mask_25_50, 'x'], vegetation_points.loc[mask_25_50, 'y'],
    bins=[x_grid, y_grid],
    density=False
)

density_50_75, _, _ = np.histogram2d(
    vegetation_points.loc[mask_50_75, 'x'], vegetation_points.loc[mask_50_75, 'y'],
    bins=[x_grid, y_grid],
    density=False
)
density_75_100, _, _ = np.histogram2d(
    vegetation_points.loc[mask_75_100, 'x'], vegetation_points.loc[mask_75_100, 'y'],
    bins=[x_grid, y_grid],
    density=False
)



total_vegetation_points, _, _ = np.histogram2d(
    vegetation_points['x'], vegetation_points['y'],
    bins=[x_grid, y_grid],
    density=False
)
points = df
total_points, _, _ = np.histogram2d(
    points['x'], points['y'],
    bins=[x_grid, y_grid],
    density=False
)

# Create rasters for each normalized density
normalized_density_overall_post = np.divide(total_vegetation_points, total_points, out=np.zeros_like(total_points), where=total_points != 0)
normalized_density_0_25_post = np.divide(density_0_25, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)
normalized_density_25_50_post = np.divide(density_25_50, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)
normalized_density_50_75_post = np.divide(density_50_75, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)
normalized_density_75_100_post = np.divide(density_75_100, total_points, out=np.zeros_like(total_vegetation_points), where=total_vegetation_points != 0)



# Display DSM, DEM, and CHM as images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(dem_post, cmap='terrain',extent=(0, scale, 0, scale))
plt.title('DEM (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')



plt.subplot(1, 2, 2)
plt.imshow(dsm_post, cmap='viridis',extent=(0, scale, 0, scale))
plt.title('DSM (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('dem1_post.png')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(chm_post, cmap='YlGnBu',extent=(0, scale, 0, scale))
plt.title('Canopy Height (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_overall_post),k=-1), cmap='viridis',extent=(0, scale, 0, scale))
plt.title('Overall Vegetation Density')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')


plt.savefig('dem2_post.png')

# Display density rasters
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.rot90(np.flipud(normalized_density_0_25_post),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (0.5 - 7.5 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_25_50_post),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (7.5 - 15 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('veg1_post.png')

plt.subplot(1, 2, 1)
plt.imshow(np.rot90(np.flipud(normalized_density_50_75_post),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (15 - 22.5 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_75_100_post),k=-1), cmap='viridis', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Vegetation Density (22.5 m +)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')


plt.savefig('veg3_post.png')





# Display DSM, DEM, and CHM as images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(dem_post-dem, cmap='seismic',extent=(0, scale, 0, scale))
plt.title('Change in DEM (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(dsm_post-dsm, cmap='seismic',extent=(0, scale, 0, scale))
plt.title('Change in DSM (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')



plt.savefig('dem1_change')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(chm_post-chm, cmap='seismic',extent=(0, scale, 0, scale))
plt.title('Change in Canopy Height (m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(normalized_density_overall_post-normalized_density_overall),k=-1), cmap='seismic',extent=(0, scale, 0, scale))
plt.title('Change in Vegetation Density')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')


plt.savefig('dem2_change')

mask_0_25 = np.logical_or(normalized_density_0_25 != 0, normalized_density_0_25_post != 0)
mask_25_50 = np.logical_or(normalized_density_25_50 != 0, normalized_density_25_50_post != 0)
mask_50_75 = np.logical_or(normalized_density_50_75 != 0, normalized_density_50_75_post != 0)
mask_75_100 = np.logical_or(normalized_density_75_100 != 0, normalized_density_75_100_post != 0)

plt.figure(figsize=(20, 10))

# Display density rasters with masks
plt.subplot(1, 2, 1)
plt.imshow(np.rot90(np.flipud(np.ma.masked_where(~mask_0_25,  normalized_density_0_25_post-normalized_density_0_25 )), k=-1),
           cmap='seismic', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Change in Vegetation Density (0.5 - 7.5 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(np.ma.masked_where(~mask_25_50, normalized_density_25_50_post-normalized_density_25_50 )), k=-1),
           cmap='seismic', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Change in Vegetation Density (7.5 - 15 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('veg1_change.png')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.rot90(np.flipud(np.ma.masked_where(~mask_50_75,  normalized_density_50_75_post-normalized_density_50_75 )), k=-1),
           cmap='seismic', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Change in Vegetation Density (15 - 22.5 m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.flipud(np.ma.masked_where(~mask_75_100, normalized_density_75_100_post-normalized_density_75_100 )), k=-1),
           cmap='seismic', interpolation='nearest',extent=(0, scale, 0, scale))
plt.title('Change in Vegetation Density (22.5+ m)')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('veg3_change.png')