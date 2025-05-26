import numpy as np
import  matplotlib.pyplot as plt
#


plt.style.use('seaborn')

fig, ax = plt.subplots(1, 3, figsize=(17, 4)) #sharex='col',
fig.subplots_adjust(bottom=0.25,wspace=0.25)

# for i in range(2):
#     for j in range(3):
#         ax[i,j].text(0.5,0.5,str((i,j)), fontsize=18, ha='center')

#first pic





labels = [ 'OUbLi', 'SISA', 'VBU', 'BFU', 'HBFU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]


# model acc
unl_on_mnist = [0.9836, 0.9853,0.7869,0.9870, 0.9810 ]
unl_on_cifar10 = [0.7654, 0.7849, 0.4323, 0.7820, 0.7780 ]
unl_on_celebA = [0.9604 , 0.9592, 0.9353, 0.9601, 0.9542 ]

for i in range(len(labels)):
    unl_on_mnist[i] = unl_on_mnist[i]*100
    unl_on_cifar10[i] = unl_on_cifar10[i]*100
    unl_on_celebA[i] = unl_on_celebA[i]*100

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# if the figure are more than 1 line, it will be ax[0,0]
ax[0].bar(x - width / 2 - width / 8 + width / 8  + width / 8, unl_on_mnist,   width=0.21, label='On MNIST', color='#9BC985', edgecolor='black', hatch='/')
ax[0].bar(x - width / 8 - width / 16 + width / 8,  unl_on_cifar10, width=0.21, label='On CIFAR10', color='#F7D58B', edgecolor='black', hatch='*')
ax[0].bar(x + width / 8 + width / 8, unl_on_celebA, width=0.21, label='On CelebA', color='#B595BF',edgecolor='black', hatch='\\')
# ax[0].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy (%)', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 121, 20)
ax[0].set_yticks(my_y_ticks )
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
# ax[0].set_title('On MNIST', fontsize=20)
# leg = ax[0,0].legend(fancybox=True, shadow=True)
# ax[0,0].legend(loc='upper left', fontsize=20)
# plt.xlabel('$\it{ESS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)








#picture 2






labels = [ 'OUbLi', 'SISA', 'VBU', 'BFU', 'HBFU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]



# model back acc
unl_on_mnist = [0.1500, 0.0967, 0.0000, 0.0916, 0.0255]
unl_on_cifar10 = [0.1320, 0.0760, 0.0000, 0.0970, 0.1804]
unl_on_celebA = [0.0500 , 0.0531, 0.0664, 0.4192, 0.8193]


x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[1].bar(x - width / 2 - width / 8 + width / 8 + width / 8, unl_on_mnist,   width=0.21, label='On MNIST', color='#9BC985', edgecolor='black', hatch='/')
ax[1].bar(x - width / 8 - width / 16 + width / 8,  unl_on_cifar10, width=0.21, label='On CIFAR10', color='#F7D58B', edgecolor='black', hatch='*')
ax[1].bar(x + width / 8 + width / 8, unl_on_celebA, width=0.21, label='On CelebA', color='#B595BF',edgecolor='black', hatch='\\')
#ax[1].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1].set_ylabel('Average UE', fontsize=20)
ax[1].set_ylabel('Backdoor Acc. (%)', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)
# ax[1].set_ylabel('Average UE', fontsize=20)
my_y_ticks = np.arange(0, 0.88, 0.2)
ax[1].set_yticks(my_y_ticks )
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
# ax[1].set_title('On CIFAR0', fontsize=20)

# leg = ax[1].legend(fancybox=True, shadow=True)

# plt.xlabel('$\it{ESS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)








#picture 3

#
# labels = [ 'OUL', 'SISA', 'VBU', 'BFU', 'HBFU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
#
#
# # model back acc
# # unl_on_mnist = [0.87358, 0.873334,0.87012, 0.8716488, 0.85341]
# # unl_on_cifar10 = [0.87385, 0.8731, 0.87280, 0.87358, 0.873049 ]
# # unl_on_celebA = [0.82807, 0.82762, 0.83108, 0.830676 , 0.83067 ]
#
# unl_on_mnist = [0.87358, 0.9999,0.9999, 0.874358, 0.875358]
# unl_on_cifar10 = [0.87385, 0.9999, 0.9999, 0.874385, 0.874385]
# unl_on_celebA = [0.82807, 0.9999, 0.9999, 0.829807 , 0.82807]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[2].bar(x - width / 2 - width / 8 + width / 8 , unl_on_mnist,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[2].bar(x - width / 8 - width / 16,  unl_on_cifar10, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[2].bar(x + width / 8, unl_on_celebA, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# # ax[2].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
#
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax[1].set_ylabel('Average UE', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[2].set_ylabel('Rec. Similarity', fontsize=20)
# ax[2].set_xticks(x)
# ax[2].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
# # ax[1].set_ylabel('Average UE', fontsize=20)
# my_y_ticks = np.arange(0, 1.21, 0.2)
# ax[2].set_yticks(my_y_ticks )
# # ax.set_yticklabels(my_y_ticks,fontsize=15)
#
# # Set the background of the axes, which is the area of the plot, to grey
# # plt.gca().set_facecolor('grey')
#
# # Set the grid with white color and a specific linestyle and linewidth
# # plt.grid(color='white', linestyle='-', linewidth=0.5)
#
#
# # plt.grid(axis='y')
# # ax[2].set_title('On CelebA', fontsize=20)
#



#picture 4


labels = [ 'OUbLi', 'SISA', 'VBU', 'BFU', 'HBFU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]



# model back acc
# unl_on_mnist = [0.87358, 0.873334,0.87012, 0.8716488, 0.85341]
# unl_on_cifar10 = [0.87385, 0.8731, 0.87280, 0.87358, 0.873049 ]
# unl_on_celebA = [0.82807, 0.82762, 0.83108, 0.830676 , 0.83067 ]

unl_on_mnist = [3.920, 11.70, 0.631, 16.03, 18.32]
unl_on_cifar10 = [6.633, 103, 0.587, 141.26, 149]
unl_on_celebA = [1.357, 133.74, 0.672, 176.86, 183]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[2].bar(x - width / 2 - width / 8 + width / 8 + width / 8, unl_on_mnist,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[2].bar(x - width / 8 - width / 16 + width / 8 ,  unl_on_cifar10, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[2].bar(x + width / 8 + width / 8 , unl_on_celebA, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[2].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1].set_ylabel('Average UE', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[2].set_ylabel('Running Time (s)', fontsize=20)
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)
# ax[1].set_ylabel('Average UE', fontsize=20)
my_y_ticks = np.arange(0, 221, 40)
ax[2].set_yticks(my_y_ticks )
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
# ax[2].set_title('On CelebA', fontsize=20)



#figure 1,0


#
# labels = ['SISA', 'VBU', 'RFU', 'HBU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
# unl_ss_in = [0.945938 , 0.949387 , 0.93905 , 0.94560 ]
# unl_ss_not_in = [0.966937 , 0.962868   , 0.95540   , 0.959398]
#
#
# unl_ms_in = [0.910400    , 0.91247 , 0.91605 , 0.9012468]
# unl_ms_not_in = [0.93619  , 0.93553 , 0.9386  , 0.92299]
#
#
# sisa_unl = [0.94898, 0.95703, 0.686785, 0.72633]
# vbu_unl = [0.95520, 0.95662, 0.683197, 0.71146]
# rfu_unl = [0.9341333, 0.93962 ,  0.679345 , 0.7261142]
# hbu_unl = [0.9510520,0.95290, 0.685824, 0.72292]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[1,0].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[1,0].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[1,0].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[1,0].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
#
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1,0].set_ylabel('Rec. Similarity', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[1,0].set_xticks(x)
# ax[1,0].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
#
# my_y_ticks = np.arange(0.76, 1.01, 0.04)
# ax[1,0].set_yticks(my_y_ticks )
# ax[1,0].set_ylim(0.76, 1.02)
# # plt.ylim()
# # ax.set_yticklabels(my_y_ticks,fontsize=15)
#
#
# # figure 1,1
#
#
#
#
# labels = ['SISA', 'VBU', 'RFU', 'HBU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
# unl_ss_in = [0.972845 , 0.973203, 0.97236328 -0.001 , 0.9731583-0.001]
# unl_ss_not_in = [0.975451 , 0.975984 , 0.9756386  -0.001  , 0.976081-0.001]
#
#
# unl_ms_in = [0.96839   ,0.969446  , 0.96787 -0.001, 0.968382 -0.001]
# unl_ms_not_in = [0.971182  , 0.972014 , 0.9715819-0.001 , 0.971294-0.001]
#
#
# sisa_unl = [0.97303, 0.973511, 0.96608, 0.967100]
# vbu_unl = [0.972769, 0.97564, 0.966605, 0.97029]
# rfu_unl = [0.971316, 0.97500,  0.9661485, 0.97007]
# hbu_unl = [0.97288, 0.976009, 0.966252,  0.970016]
#
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[1,1].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[1,1].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[1,1].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[1,1].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
#
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax[1,1].set_ylabel('Rec. Similarity', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[1,1].set_xticks(x)
# ax[1,1].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
#
# my_y_ticks = np.arange(0.5, 2.1, 0.02)
# ax[1,1].set_yticks(my_y_ticks )
# ax[1,1].set_ylim(0.9,1.02)
#
#
#
#
# #figure 1,2
#
#
#
# labels = ['SISA', 'VBU', 'RFU', 'HBU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
# unl_ss_in = [0.97557 , 0.976940, 0.97633 -0.001, 0.977246 -0.001]
# unl_ss_not_in = [0.97833   , 0.97978  , 0.979787   -0.001 , 0.98027489-0.001]
#
#
# unl_ms_in = [0.968087   , 0.9652563  , 0.967717-0.001, 0.9661944-0.001]
# unl_ms_not_in = [0.97242   , 0.97291337 , 0.973726706 -0.001 , 0.972056-0.001]
#
#
# sisa_unl = [0.977335, 0.979339, 0.96394, 0.9669511]
# vbu_unl = [0.97762, 0.98005, 0.9641662, 0.96662]
# rfu_unl = [0.964420, 0.966258, 0.963810, 0.9661702]
# hbu_unl = [0.977701, 0.979951, 0.965032,  0.9672170]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[1,2].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[1,2].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[1,2].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[1,2].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
#
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax[1,1].set_ylabel('Rec. Similarity', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[1,2].set_xticks(x)
# ax[1,2].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
#
# my_y_ticks = np.arange(0.5, 2.1, 0.02)
# ax[1,2].set_yticks(my_y_ticks )
# ax[1,2].set_ylim(0.9,1.02)



#figure 2,0


#
# labels = ['SISA', 'VBU', 'RFU', 'HBU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
# unl_ss_in = [0.3100  , 0.2700 , 0.2837 , 0.2603 ]
# unl_ss_not_in = [0.9933  , 0.9907 , 0.9957     , 0.9923]
#
#
# unl_ms_in = [0.2020    , 0.1833  , 0.1737, 0.1707]
# unl_ms_not_in = [0.9583   , 0.9597 , 0.9607  , 0.9003]
#
# mib_ms_in = [0,0,0,0]
# mib_ms_not_in = [1,1,1,1]
#
# sisa_unl = [0.1513, 0.9963, 0.0010, 0.8770]
# vbu_unl = [0.2593, 0.9953, 0.0273, 0.5890]
# rfu_unl = [0.2817, 0.9947,  0.0023, 0.7283]
# hbu_unl = [0.2170, 0.9927, 0.0077, 0.7910]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[2,0].bar(x - width /6 - width / 6  , unl_ss_in,   width=width/6, label='PEDR SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[2,0].bar(x - width / 6,  unl_ss_not_in, width=width/6, label='PEDR SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[2,0].bar(x  , unl_ms_in, width=width/6, label='PEDR MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[2,0].bar( x + width / 6 , unl_ms_not_in, width=width/6, label='PEDR MS Not In', color='#797BB7', edgecolor='black', hatch='x')
# ax[2,0].bar(x + width / 6 + width/6 , mib_ms_in, width=width/6, label='MIB B-MS In', color='#9CD1C8', edgecolor='black', hatch='o')
#
#
# ax[2,0].bar(x + width / 6 + width / 6 + width/6  , mib_ms_not_in,   width=width/6, label='MIB B-MS Not In', color='#E58579', edgecolor='black', hatch='\\')
#
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[2,0].set_ylabel('Verifiability', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[2,0].set_xticks(x)
# ax[2,0].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
#
# my_y_ticks = np.arange(0, 1.5, 0.2)
# ax[2,0].set_yticks(my_y_ticks )
# # ax[2,0].set_ylim(0.5,1.2)
# # plt.ylim()
# # ax.set_yticklabels(my_y_ticks,fontsize=15)
#
#
#
#
# #figure 2,1
#
#
#
# labels = ['SISA', 'VBU', 'RFU', 'HBU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
# unl_ss_in = [0.0708 , 0.0528 , 0.0516 , 0.0760]
# unl_ss_not_in = [0.9564  , 0.9752  , 0.9708   , 0.9580]
#
#
# unl_ms_in = [0.0628   , 0.0096 ,  0.0112 , 0.0188]
# unl_ms_not_in = [0.9468  , 0.8708, 0.9088  , 0.9196]
#
#
# sisa_unl = [0.1204, 0.9060, 0.0604, 0.8648]
# vbu_unl = [0.0752, 0.9704, 0.0228, 0.9192]
# rfu_unl = [ 0.0396, 0.9428,  0.0260, 0.9280]
# hbu_unl = [0.0568, 0.9404, 0.0164, 0.9116]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[2,1].bar(x - width /6 - width / 6  , unl_ss_in,   width=width/6, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[2,1].bar(x - width / 6,  unl_ss_not_in, width=width/6, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[2,1].bar(x  , unl_ms_in, width=width/6, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[2,1].bar( x + width / 6 , unl_ms_not_in, width=width/6, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
# ax[2,1].bar(x + width / 6 + width/6 , mib_ms_in, width=width/6, label='MIB B-MS In', color='#9CD1C8', edgecolor='black', hatch='o')
#
#
# ax[2,1].bar(x + width / 6 + width / 6 + width/6  , mib_ms_not_in,   width=width/6, label='MIB B-MS Not In', color='#E58579', edgecolor='black', hatch='\\')
#
#
# # plt.bar(x - width /6 - width / 6 , unl_muv_MNIST, width=width/6, label='MUV MNIST', color='#C6B3D3', edgecolor='black', hatch='/')
# #
# # plt.bar(x - width / 6 , unl_muv_CIFAR, width=width/6,  label='MUV CIFAR10', color='#F1DFA4', edgecolor='black' , hatch='x')
# # plt.bar(x , unl_muv_CelebA, width=width/6, label='MUV CelebA', color='#80BA8A', edgecolor='black', hatch='o')
# #
# #
# # plt.bar(x + width / 6  , unl_mib_MNIST,   width=width/6, label='MIB MNIST', color='#9CD1C8', edgecolor='black',  hatch='-')
# #
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax[2,0].set_ylabel('Verifiability', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[2,1].set_xticks(x)
# ax[2,1].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
#
# my_y_ticks = np.arange(0, 1.5, 0.2)
# ax[2,1].set_yticks(my_y_ticks )
# # ax[2,0].set_ylim(0.5,1.2)
# # plt.ylim()
# # ax.set_yticklabels(my_y_ticks,fontsize=15)
#
#
# #figure 2, 2
#
#
#
# labels = ['SISA', 'VBU', 'RFU', 'HBU']
# #unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]
#
# unl_ss_in = [0.1591  , 0.1496 , 0.1762 , 0.1445 ]
# unl_ss_not_in = [0.9314   , 0.9703   , 0.9611    , 0.9611]
#
#
# unl_ms_in = [0.0984     , 0.1055 , 0.0994 , 0.0881]
# unl_ms_not_in = [0.9262  , 0.9395, 0.9365  , 0.9098]
#
#
# sisa_unl = [0.1855, 0.9682, 0.0881, 0.9170]
# vbu_unl = [0.1506, 0.9447,  0.1107, 0.9549]
# rfu_unl = [0.1383, 0.9549,  0.1383, 0.9693]
# hbu_unl = [0.1619, 0.9518, 0.1148, 0.9529]
#
#
# x = np.arange(len(labels))  # the label locations
# width = 0.7  # the width of the bars
# # no_noise = np.around(no_noise,0)
# # samping = np.around(samping,0)
# # ldp = np.around(ldp,0)
#
# # plt.style.use('bmh')
#
#
#
#
# #plt.subplots(figsize=(8, 5.3))
# #plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
# ax[2,2].bar(x - width /6 - width / 6  , unl_ss_in,   width=width/6, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
# ax[2,2].bar(x - width / 6,  unl_ss_not_in, width=width/6, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
# ax[2,2].bar(x  , unl_ms_in, width=width/6, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
# ax[2,2].bar( x + width / 6 , unl_ms_not_in, width=width/6, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
# ax[2,2].bar(x + width / 6 + width/6 , mib_ms_in, width=width/6, label='MIB B-MS In', color='#9CD1C8', edgecolor='black', hatch='o')
#
#
# ax[2,2].bar(x + width / 6 + width / 6 + width/6  , mib_ms_not_in,   width=width/6, label='MIB B-MS Not In', color='#E58579', edgecolor='black', hatch='\\')
#
#
#
# # plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# # plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# # plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax[2,0].set_ylabel('Verifiability', fontsize=20)
# # ax.set_title('Performance of Different Users n')
# ax[2,2].set_xticks(x)
# ax[2,2].set_xticklabels(labels ,fontsize=13)
# # ax.set_xticklabels(labels,fontsize=15)
#
# my_y_ticks = np.arange(0, 1.5, 0.2)
# ax[2,2].set_yticks(my_y_ticks )
# ax[2,0].set_ylim(0.5,1.2)
# plt.ylim()
# ax.set_yticklabels(my_y_ticks,fontsize=15)



handles, labels = ax[1].get_legend_handles_labels()
# Create a "dummy" handle for the legend title
#title_handle = plt.Line2D([], [], color='none', label='Method')
title_handle = plt.Line2D([], [], color='none', label='Method')

# Insert the title handle at the beginning of the handles list
handles = [title_handle] + handles
# handles.insert(1, title_handle)
labels = ['Datasets:'] + labels
# labels.insert(1, '')

# both are 1, so it can be consistent


fig.legend(handles, labels, frameon=True, facecolor='#EAEAF2', loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.0),fontsize=18)

# plt.legend( title = 'Methods and Datasets',frameon=True, facecolor='white', loc='best',
#            ncol=2, mode="expand", framealpha=0.5, borderaxespad=0., fontsize=20, title_fontsize=20)

# fig.tight_layout()
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('test_06_all.pdf', dpi=200, bbox_inches='tight')

plt.show()