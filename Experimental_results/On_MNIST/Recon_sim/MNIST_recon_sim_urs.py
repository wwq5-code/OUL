

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5 ]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['500', '1000', '1500', '2000', '2500']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL = [0.54456, 0.563, 0.576, 0.559,  0.574 ]

OUbLi_w = [0.779, 0.716, 0.685, 0.669, 0.659 ]
org_acc = [0.999, 0.999, 0.999, 0.999, 0.999 ]

org = [1, 1, 1, 1, 1 ]

vbu_acc = [0.999, 0.999, 0.999, 0.999, 0.999 ]
#vbu_acc = [0.870840, 0.8682785, 0.8606031, 0.86109, 0.8622438, 0.861085]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]

vbu_ldp_acc = [0.789, 0.732, 0.693, 0.676, 0.668 ]

for i in range(len(OUL)):
    OUL[i] = OUL[i]*1
    org_acc[i] = org_acc[i]
    vbu_acc[i] = vbu_acc[i]*1
    vbu_ldp_acc[i] = vbu_ldp_acc[i]*1

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, OUbLi_w, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='OUbLi (w)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='OUbLi', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)


plt.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
         label='VBU (No Pri.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, vbu_ldp_acc, linestyle='-.', color='#E07B54',  marker='*', fillstyle='full', markevery=markevery,
         label='BFU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s) ##E1C855


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Reconstruction Similarity' ,fontsize=24)
my_y_ticks = np.arange(0.50, 1.04, 0.1)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\\it USS$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

#plt.annotate(r"1e-1", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc=(0.3, 0.4),fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_recon_sim_urs.pdf', format='pdf', dpi=200)
plt.show()