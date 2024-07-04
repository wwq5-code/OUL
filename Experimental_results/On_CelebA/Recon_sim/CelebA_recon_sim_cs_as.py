

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]
OUL = [0.826, 0.823, 0.819, 0.821, 0.8200, 0.8178799]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL_know = [0.831, 0.831, 0.827, 0.829, 0.83308897, 0.83308799]

org_acc = [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999]

vbu_acc = [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999]
# vbu_acc = [0.824204, 0.8264, 0.8287212, 0.82834, 0.833447, 0.82569]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]


vbu_ldp_acc = [0.9655618, 0.9655618, 0.9655618, 0.9655618, 0.9655618, 0.9655618]


for i in range(len(OUL)):
    OUL[i] = OUL[i]*1
    org_acc[i] = org_acc[i]*1
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
plt.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='OUbL', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
#plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery, label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
plt.plot(x, OUL_know, linestyle='-.', color='#E07E35',  marker='p', fillstyle='full', markevery=markevery,
         label='OUbL (Know Unl. Int.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



# plt.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
#          label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
# plt.plot(x, vbu_ldp_acc, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
#          label='VBU-LDP',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Reconstruction Similarity' ,fontsize=24)
my_y_ticks = np.arange(0.80, 1.01, 0.04)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\it{CSR}$ and $\it{ASR}$ (%)',fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

#plt.annotate(r"1e-1", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('CelebA_recon_sim_cs_as.pdf', format='pdf', dpi=200)
plt.show()