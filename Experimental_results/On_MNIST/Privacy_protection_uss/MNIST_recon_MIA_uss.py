

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['500', '1000', '1500', '2000', '2500']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL = [0.5020, 0.5010 , 0.5007, 0.5010, 0.5006]

OUbLi_w = [0.6410, 0.6135, 0.6047, 0.5980, 0.5996]

bfu_acc = [0.6490, 0.6200, 0.6009, 0.6075, 0.6070]

vbu_acc = [0.9999, 0.9999, 0.9999, 0.9999,0.9999]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]
bfu_ldp_acc = [0.78531, 0.78531, 0.78531, 0.78531, 0.78531]


for i in range(len(OUL)):
    OUL[i] = OUL[i]*1
    OUbLi_w[i] = OUbLi_w[i]*1
    vbu_acc[i] = vbu_acc[i]*1
    bfu_ldp_acc[i] = bfu_ldp_acc[i]*1

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 4.1))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)

#plt.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery, label='VBU (No Pri.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, OUbLi_w, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='OUbLi (w)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='OUbLi', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
#plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery, label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, bfu_acc, linestyle=':', color='#E07B54',  marker='*', fillstyle='full', markevery=markevery,
         label='BFU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, bfu_ldp_acc, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,label='BFU-DP',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MIA Accuracy (%)' ,fontsize=20)
my_y_ticks = np.arange(0.45, 0.701, 0.05)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel(r'$\it{USS}$',fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

#plt.annotate(r"1e-1", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc=[0.02,0.21],fontsize=18)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_recon_mia_uss.pdf', format='pdf', dpi=200)
plt.show()