import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from astropy.convolution import Gaussian2DKernel

#gaussian_2D_kernel = Gaussian2DKernel(10,50,np.pi,x_size=61,y_size=61)
#print(np.shape(gaussian_2D_kernel.array))
#plt.imshow(gaussian_2D_kernel, interpolation='none', origin='lower')
#plt.xlabel('x [pixels]')
#plt.ylabel('y [pixels]')
#plt.colorbar()
#plt.show()



# test on real micrograph
import mrcfile
map_name="test_micrograph/test_micrograph.mrc"
mrc_data = mrcfile.open(map_name, 'r+')
v_size=mrc_data.voxel_size
nx=mrc_data.header['nx']
ny=mrc_data.header['ny']
nz=mrc_data.header['nz']
apix=v_size['z']
data=mrc_data.data


# simulate with single staright filament
data=np.zeros((1000,1000))
data[200:800,300:305]=1

# shrink micrograph
#shrink = (slice(0, None, 2), slice(0, None, 2))
#data=data[shrink]


print(np.shape(data))
data_shape=np.shape(data)

fdata=np.memmap("micrograph.npy",dtype="float32",mode="w+",shape=data_shape)
fdata[:]=data[:]
fdata.flush()
data=np.memmap("micrograph.npy",dtype="float32",mode="r",shape=data_shape)


# initialize final plot
resp_ratio=np.zeros(data_shape)
resp_orient=np.zeros(data_shape)

print(np.shape(resp_ratio))
print(np.shape(resp_orient))

kernel_size=61
for i in range(0,9):
	theta_1=i/18
	theta_2=theta_1+1/2
	kernel_1 = Gaussian2DKernel(10,50,theta_1*np.pi,x_size=kernel_size,y_size=kernel_size).array
	kernel_2 = Gaussian2DKernel(10,50,theta_2*np.pi,x_size=kernel_size,y_size=kernel_size).array
	kernel_shape=np.shape(kernel_1)
	print(np.shape(kernel_1))

	fig_1, axes_1 = plt.subplots(2,3,figsize=(10,8),dpi=160)
	plt.gray()

	axes_1[0][0].imshow(data)
	print(data_shape)

	axes_1[0][1].imshow(np.real(kernel_1))
	axes_1[0][2].imshow(np.real(kernel_2))
	axes_1[1][0].imshow(data)
	resp_1=ndi.convolve(data, kernel_1, mode='wrap')
	resp_2=ndi.convolve(data, kernel_2, mode='wrap')
	axes_1[1][1].imshow(resp_1)
	axes_1[1][2].imshow(resp_2)

	#plt.show()
	plt.savefig('%d_%d_1.png'%(i,kernel_size))
	plt.close()

	fig_2, axes_2 = plt.subplots(1,3,figsize=(10,8),dpi=160)
	plt.magma()
	diff_1=np.nan_to_num(np.divide(resp_1,resp_2),nan=0)
	diff_2=np.nan_to_num(np.divide(resp_2,resp_1),nan=0)
	
	tmp_1=np.zeros(data_shape)
	tmp_1=np.multiply(diff_1,diff_1>=diff_2)
	tmp_2=np.zeros(data_shape)
	tmp_2=np.multiply(diff_2,diff_1<diff_2)
	diff=tmp_1+tmp_2
	
	
	axes_2[0].imshow(diff)
	axes_2[1].imshow(diff_1)
	axes_2[2].imshow(diff_2)
	#plt.show()
	plt.savefig('%d_%d_2.png'%(i,kernel_size))
	plt.close()
	
	print(np.min(diff),np.max(diff))
	print(np.min(diff_1),np.max(diff_1))
	print(np.min(diff_2),np.max(diff_2))
	
	tmp_ratio_1=np.zeros(data_shape)
	tmp_orient_1=np.multiply(np.ones(data_shape),theta_1*np.pi)
	tmp_ratio_1=np.multiply(diff,diff>=resp_ratio)
	tmp_orient_1=np.multiply(tmp_orient_1,diff>=resp_ratio)
	
	tmp_ratio_2=np.zeros(data_shape)
	tmp_orient_2=np.zeros(data_shape)
	tmp_ratio_2=np.multiply(resp_ratio,diff<resp_ratio)
	tmp_orient_2=np.multiply(resp_orient,diff<resp_ratio)
	
	resp_ratio=tmp_ratio_1+tmp_ratio_2
	resp_orient=tmp_orient_1+tmp_orient_2
	

fig_2, axes_2 = plt.subplots(1,2,figsize=(10,8),dpi=160)
#plt.gray()
# show in log scale
axes_2[0].imshow(np.log(resp_ratio))
axes_2[1].imshow(resp_orient)
	
plt.show()

