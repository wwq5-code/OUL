
import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5 ]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['500', '1000', '1500', '2000', '2500' ]
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL = [0.9604, 0.9643, 0.9635, 0.9630, 0.9627]

org_acc = [0.9605, 0.9617, 0.9621, 0.9635, 0.9641 ]

vbu_acc = [0.7869, 0.7869, 0.7869, 0.7869, 0.7869, 0.7869]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]
Shadow_m = [0.8572, 0.9055, 0.9158, 0.9253, 0.9319]

for i in range(len(OUL)):
    OUL[i] = OUL[i]*100
    org_acc[i] = org_acc[i]*100
    vbu_acc[i] = vbu_acc[i]*100
    Shadow_m[i] = Shadow_m[i]*100

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Original Model',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='OUbLi', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
#plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery, label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, Shadow_m, linestyle=':', color='#E07B54',  marker='*', fillstyle='full', markevery=markevery,
         label='Shadow Model',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


#
# plt.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
#          label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
# plt.plot(x, vbu_ldp_acc, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
#          label='VBU-LDP',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#

# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Test Accuracy (%)' ,fontsize=24)
my_y_ticks = np.arange(80., 100.1, 4)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\it{ASS}$ and $\it{CSS}$',fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

# plt.annotate(r"1e0", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc=(0.3, 0.2),fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('CelebA_model_acc_cs_as.pdf', format='pdf', dpi=200)
plt.show()