

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
OUL = [1.0, 1.0, 0.99833, 0.99667, 0.995, 1.0, 1.0, 1.0, 1.0, 0.99333, 0.99167, 0.875, 0.75833, 0.63167, 0.46667, 0.33333, 0.275, 0.245, 0.2321, 0.2321, 0.232, 0.2255, 0.2285, 0.2231167, 0.2295, 0.228333, 0.224833, 0.224167, 0.21333, 0.15, 0.125, 0.12833, 0.13, 0.13333, 0.13667,0.10333, 0.115, 0.125, 0.119833, 0.120833, 0.1185, 0.10333, 0.08167, 0.075, 0.085, 0.09333, 0.095, 0.095, 0.095, 0.09667, 0.09667, 0.1, 0.10833, 0.10833, 0.105, 0.10167, 0.09667, 0.09667, 0.09833, 0.09667, 0.095, 0.095, 0.095, 0.09333, 0.095, 0.095, 0.095, 0.09167, 0.09167, 0.09167, 0.09, 0.09333,0.09333, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.09667, 0.09667, 0.09667, 0.09667, 0.09833, 0.10333, 0.10833, 0.11167, 0.115, 0.10833, 0.11167, 0.10667, 0.10833, 0.10667, 0.1, 0.09667, 0.095, 0.095, 0.09333, 0.09333, 0.09167, 0.09333, 0.095, 0.095, 0.095, 0.09333, 0.09333, 0.09333, 0.095,0.09333, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.09667, 0.09667, 0.09667, 0.09667, 0.09833, 0.10333, 0.10833, 0.11167, 0.115, 0.10833, 0.11167, 0.10667, 0.10833, 0.10667, 0.1, 0.09667, 0.095, 0.095, 0.09333, 0.09333, 0.09167, 0.09333, 0.095, 0.095, 0.095, 0.09333, 0.09333, 0.09333, 0.095]

clean_sample = [1.0, 0.99833, 0.99167, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99667, 0.9935, 0.955, 0.99, 0.99833, 0.99833, 0.99833, 0.99833, 0.99833, 0.99833, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99833, 0.995, 0.98833, 0.905, 0.983333, 0.9835, 0.988333, 0.97667, 0.99167, 0.99667, 1.0, 1.0,1.0, 0.98167, 0.986167, 0.9795, 0.9795, 0.974667, 0.974833, 0.979, 0.983333, 0.988333, 0.9905, 0.9935, 0.996333, 0.99667, 0.99667, 1.0, 1.0, 1.0, 0.99833, 0.995, 0.99333, 0.99333, 0.99167, 0.99167, 0.99, 0.98833, 0.98667, 0.96833, 0.992667, 0.989167, 0.986667, 0.983833, 0.981333, 0.979333, 0.980667, 0.984333, 0.987667, 0.990333, 0.990667, 0.981167, 0.968833, 0.967167, 0.972833, 0.979, 0.983667, 0.989333, 0.95167, 0.925, 0.988833, 0.9905, 0.9789833, 0.988667, 0.988167, 0.91, 0.92833, 0.94167, 0.95, 0.96, 0.96333, 0.96333, 0.96, 0.95167, 0.94833, 0.93, 0.9505, 0.915, 0.93833, 0.94833, 0.97333, 0.98167, 0.98333, 0.98833, 0.98833, 0.98833, 0.99, 0.99167,0.99333, 0.96833, 0.9, 0.93875, 0.89167, 0.96167, 0.99333, 0.99833, 1.0, 1.0, 1.0, 0.995, 0.94167, 0.74667, 0.64, 0.55667, 0.50667, 0.55667, 0.63333, 0.71667, 0.785, 0.85167, 0.89167, 0.94667, 0.98167, 0.99333, 0.99333, 0.99333, 0.99667, 0.99667, 0.99667, 0.99333, 0.985, 0.97833, 0.96333, 0.95, 0.93, 0.91]


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
plt.figure(figsize=(5., 3.8))
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


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc='best',fontsize=16)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_acc_during_training.pdf', format='pdf', dpi=200)
plt.show()