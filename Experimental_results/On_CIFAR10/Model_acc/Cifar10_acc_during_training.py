

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
OUL = [0.956, 0.5, 0.014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.008, 0.018, 0.034, 0.06, 0.082, 0.013, 0.0194, 0.035, 0.0442, 0.0388, 0.0276, 0.0164, 0.012, 0.0092, 0.07, 0.056, 0.05, 0.04, 0.042, 0.052, 0.04, 0.058, 0.058, 0.068, 0.092, 0.096, 0.09, 0.094, 0.092, 0.088, 0.1, 0.108, 0.128, 0.136, 0.12, 0.106, 0.082, 0.082, 0.076, 0.072, 0.082, 0.102, 0.102, 0.106, 0.1, 0.11, 0.112, 0.124, 0.112, 0.084, 0.092, 0.058, 0.054, 0.078, 0.104, 0.086, 0.082, 0.074, 0.078, 0.086, 0.076, 0.064, 0.048, 0.056, 0.08, 0.11, 0.158, 0.174, 0.182, 0.14, 0.11, 0.094, 0.076, 0.062, 0.066, 0.064, 0.064, 0.062, 0.058, 0.07, 0.074, 0.084, 0.106, 0.094, 0.066, 0.08, 0.108, 0.148, 0.176, 0.162, 0.136, 0.122, 0.074, 0.064, 0.06, 0.054, 0.06, 0.064, 0.066, 0.072, 0.076, 0.074, 0.078, 0.08, 0.08, 0.084, 0.1, 0.112, 0.11, 0.098, 0.094, 0.086, 0.084, 0.098, 0.106, 0.09, 0.082, 0.08, 0.07, 0.05, 0.042, 0.036, 0.04, 0.042, 0.052, 0.066, 0.102, 0.114, 0.12, 0.128, 0.096, 0.078, 0.056, 0.05, 0.044, 0.042, 0.044, 0.044, 0.058, 0.07, 0.074, 0.082, 0.088, 0.092, 0.078, 0.064, 0.054, 0.058, 0.068, 0.074, 0.086, 0.102, 0.114, 0.128, 0.146, 0.148, 0.126, 0.098, 0.088, 0.078, 0.074, 0.062, 0.054, 0.044, 0.044, 0.038, 0.034, 0.036, 0.046, 0.064, 0.068, 0.07, 0.076, 0.09]

clean_sample = [0.99, 0.988, 0.994, 0.998, 0.99, 0.988, 0.984, 0.984, 0.98, 0.98, 0.988, 0.986, 0.986, 0.986, 0.988, 0.992, 0.994, 0.994, 0.992, 0.988, 0.986, 0.982, 0.984, 0.986, 0.984, 0.984, 0.982, 0.98, 0.982, 0.976, 0.982, 0.98, 0.984, 0.986, 0.974, 0.988, 0.994, 0.996, 0.996, 0.996, 0.988, 0.98, 0.974, 0.966, 0.946, 0.946, 0.954, 0.96, 0.96, 0.974, 0.978, 0.984, 0.984, 0.99, 0.994, 0.994, 0.988, 0.988, 0.988, 0.984, 0.968, 0.964, 0.962, 0.948, 0.946, 0.982, 0.994, 0.99, 0.992, 0.99, 0.99, 0.966, 0.956, 0.954, 0.958, 0.98, 0.99, 0.988, 0.986, 0.984, 0.984, 0.982, 0.978, 0.974, 0.962, 0.952, 0.954, 0.958, 0.962, 0.97, 0.974, 0.974, 0.976, 0.966, 0.962, 0.952, 0.95, 0.972, 0.984, 0.98, 0.982, 0.974, 0.972, 0.96, 0.956, 0.952, 0.95, 0.954, 0.96, 0.974, 0.98, 0.982, 0.984, 0.978, 0.966, 0.948, 0.936, 0.92, 0.928, 0.938, 0.942, 0.946, 0.942, 0.944, 0.962, 0.972, 0.98, 0.974, 0.98, 0.958, 0.924, 0.92, 0.926, 0.93, 0.928, 0.926, 0.916, 0.924, 0.926, 0.932, 0.952, 0.976, 0.974, 0.984, 0.984, 0.974, 0.96, 0.956, 0.95, 0.952, 0.942, 0.944, 0.954, 0.96, 0.964, 0.968, 0.968, 0.964, 0.962, 0.958, 0.956, 0.946, 0.926, 0.948, 0.938, 0.94, 0.944, 0.948, 0.95, 0.938, 0.94, 0.94, 0.938, 0.958, 0.972, 0.98, 0.984, 0.984, 0.982, 0.98, 0.976, 0.974, 0.968, 0.964, 0.96, 0.958, 0.95, 0.954, 0.964, 0.974, 0.974, 0.976]


org_acc = [0.9850, 0.9887, 0.9876, 0.9860, 0.9855]

vbu_acc = [0.8556, 0.8256, 0.7813, 0.7933, 0.7869]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]
vbu_ldp_acc = [0.4435, 0.4747, 0.4627, 0.4357, 0.4664]

x = []
out_print = []
clean_print = []
t_i =3
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


plt.title('(b) On CIFAR10', fontsize=20)
plt.legend(loc='best',fontsize=16)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar10_acc_during_training.pdf', format='pdf', dpi=200)
plt.show()