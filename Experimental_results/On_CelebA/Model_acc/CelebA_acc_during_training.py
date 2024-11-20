

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.2', '0.4', '0.6', '0.8', '1']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL = [0.54121, 0.09348, 0.06765, 0.05043, 0.05166, 0.05289, 0.151402, 0.12423, 0.05043, 0.07872,0.09717, 0.08856, 0.08487, 0.05289, 0.07503, 0.05781, 0.14022, 0.05166, 0.10209, 0.08733, 0.08487, 0.10209, 0.08241, 0.0492, 0.06273, 0.06027, 0.06888, 0.08118, 0.08979, 0.06765, 0.06396, 0.11193, 0.05535, 0.0738, 0.0984, 0.0984, 0.07134, 0.1353, 0.08241, 0.05043, 0.04797, 0.07626, 0.05289, 0.12177, 0.05412, 0.04797, 0.05289, 0.06765, 0.05043, 0.06765, 0.05658, 0.09594, 0.05166, 0.11562, 0.07011, 0.05658, 0.06642, 0.06027, 0.06519, 0.11685, 0.10455, 0.05781, 0.11808, 0.08979, 0.06519,0.0738, 0.0615, 0.09963, 0.05043, 0.05289]


clean_sample = [0.76753, 0.75523, 0.77491, 0.71218, 0.73678, 0.77245, 0.75646, 0.73924, 0.73678, 0.72325, 0.67897, 0.6556, 0.66913, 0.65068, 0.62362, 0.644034, 0.650185, 0.659533, 0.658426, 0.644895, 0.646002, 0.646986, 0.648708, 0.649323, 0.647109, 0.648831, 0.642558, 0.634071, 0.638868, 0.643788, 0.643296, 0.620541, 0.611931, 0.607626, 0.619803, 0.631857, 0.620049, 0.622509, 0.638622, 0.617958, 0.607995, 0.617097, 0.61476, 0.629028, 0.609102, 0.627306, 0.626937, 0.619926, 0.612546, 0.631734, 0.649569, 0.63075, 0.608118, 0.619557, 0.610578, 0.631119, 0.645633, 0.639114, 0.619188, 0.609594]

org_acc = [0.9850, 0.9887, 0.9876, 0.9860, 0.9855]

vbu_acc = [0.8556, 0.8256, 0.7813, 0.7933, 0.7869]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]
vbu_ldp_acc = [0.4435, 0.4747, 0.4627, 0.4357, 0.4664]

x = []
out_print = []
clean_print = []
t_i =1
for i in range(41):
    out_print.append(OUL[i*t_i]*100)

    clean_print.append(clean_sample[i*t_i]*100)
    # org_acc[i] = org_acc[i]*100
    # vbu_acc[i] = vbu_acc[i]*100
    # vbu_ldp_acc[i] = vbu_ldp_acc[i]*100
    x.append(i)

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 4.1))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, clean_print, linestyle='-', color='#E07B54',  fillstyle='full', markevery=markevery,
         label='Clean Data',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, out_print, linestyle='-', color='#797BB7',  fillstyle='full', markevery=markevery,
         label='Synthesized Data (OUbL)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# marker='o',  marker='*',
#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
# plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
#          label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
#
#
# plt.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
#          label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
# plt.plot(x, vbu_ldp_acc, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
#          label='VBU-LDP',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Backdoor Accuracy (%)' ,fontsize=20)
my_y_ticks = np.arange(0., 101, 20)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('Training Rounds', fontsize=20)

my_x_ticks = np.arange(0, 41, 10)
plt.xticks(my_x_ticks, fontsize=20)
# plt.title('CIFAR10 IID')

# plt.annotate(r"1e0", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


plt.title('(c) On CelebA', fontsize=20)
plt.legend(loc='best',fontsize=16)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('CelebA_acc_during_training.pdf', format='pdf', dpi=200)
plt.show()