import matplotlib.pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman')
task_name_list = ['mrpc', 'qnli', 'qqp', 'sst-2']
fig,axes=plt.subplots(2,2)
for task,ax in zip(task_name_list,axes.flatten()):
	relation_matrix = np.load('figures/ralation_matrices/{}.npy'.format(task))
	im=ax.imshow(relation_matrix, cmap="YlGnBu")
	ax.set_title(task.upper(),fontdict={'size':12})
	ax.tick_params(labelsize=8,width=1,length=1,direction='in')
cb=fig.colorbar(im,ax=axes.flatten())
cb.ax.tick_params(labelsize=8,width=1,length=1,direction='out')
plt.savefig('figures/relation_matrices.eps', dpi=600, format='eps')
