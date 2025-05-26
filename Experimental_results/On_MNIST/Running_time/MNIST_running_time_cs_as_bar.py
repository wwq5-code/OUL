import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['500', '1000', '1500', '2000', '2500'  ]
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]



unl_muv_MNIST = [3.92063, 7.17106  , 9.923 , 13.39719, 16.33203 ]
unl_mib_MNIST = [3.92063 , 7.17106 , 9.923 , 13.39719, 16.33203 ]

unl_muv_CIFAR = [6.663723, 9.9689  , 14.737 , 18.6452 , 24.2110 ]
unl_mib_CIFAR = [6.66372 , 9.9689 , 14.737 , 18.6452 , 24.2110 ]

unl_muv_CelebA = [1.35, 4, 6.5, 10.8, 14.6 ]
unl_mib_CelebA = [1.35, 4, 6.5, 10.8, 14.6 ]



for i in range(len(labels)):
    unl_muv_MNIST[i] = unl_muv_MNIST[i]
    unl_mib_MNIST[i] = unl_mib_MNIST[i]
    unl_muv_CIFAR[i] = unl_muv_CIFAR[i]
    unl_mib_CIFAR[i] = unl_mib_CIFAR[i]
    unl_muv_CelebA[i] = unl_muv_CelebA[i]
    unl_mib_CelebA[i] = unl_mib_CelebA[i]


x = np.arange(len(labels))  # the label locations
width = 0.9 # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

plt.style.use('seaborn')
plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width /6 - width / 6 , unl_muv_MNIST, width=width/6, label='MUA-MD MNIST', color='#C6B3D3', edgecolor='black', hatch='/')

#plt.bar(x - width / 6 , unl_muv_CIFAR, width=width/6,  label='MUA-MD CIFAR10', color='#F7D58B', edgecolor='black' , hatch='x')
#E58579, 80BA8A


# F7D58B, 9CD1C8, C6B3D3, E58579
plt.bar(x - width / 4  , unl_mib_MNIST,   width=width/4, label='OUbLi on MNIST', color='#C6B3D3', edgecolor='black',  hatch='x')

# F7D58B , 6BB7CA
plt.bar(x , unl_mib_CIFAR, width=width/4, label='OUbLi on CIFAR10', color='#F7D58B', edgecolor='black', hatch='*')

plt.bar(x + width / 4 , unl_muv_CelebA, width=width/4, label='OUbLi on CelebA', color='#E58579', edgecolor='black', hatch='o')


#plt.bar(x + width / 6 + width / 6 + width/6  , unl_mib_CelebA,   width=width/6, label='MIB CelebA', color='#E58579', edgecolor='black', hatch='\\')
# plt.bar(x - width / 8 - width / 16, unl_vib, width=0.168, label='PriMU$_{w}$', color='cornflowerblue', hatch='*')
# plt.bar(x + width / 8, unl_self_r, width=0.168, label='PriMU$_{w/o}$', color='g', hatch='x')
# plt.bar(x + width / 2 - width / 8 + width / 16, unl_hess_r, width=0.168, label='HBFU', color='orange', hatch='\\')


# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 30, 6)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)
# plt.grid(axis='y')
# plt.legend(loc='upper left', fontsize=20)
plt.legend( frameon=True, facecolor='#EAEAF2', loc='best', bbox_to_anchor=(1.05001, -0.15),
           ncol=3, fontsize=14.6,)

# mode="expand",  columnspacing=1.0,  borderaxespad=0., framealpha=0.5,handletextpad=0.5
#title = 'Methods and Datasets',

plt.xlabel('$\it{ASS}$ and $\it{CSS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_running_time_cs_as_bar.pdf', format='pdf', dpi=200)
plt.show()
