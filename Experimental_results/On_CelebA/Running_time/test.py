import matplotlib.pyplot as plt
import numpy as np

# Example data
USR = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
running_time_origin = np.array([250, 255, 252, 256, 258, 257])
running_time_oul = np.array([40, 42, 41, 43, 44, 45])
running_time_vbu = np.array([30, 32, 31, 33, 34, 35])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

# Upper plot (y-axis range: 200-300)
ax1.plot(USR, running_time_origin, label='Origin (No Pri.)', linestyle='--', marker='s', color='lightgreen')
ax1.plot(USR, running_time_oul, label='OUL', linestyle='-', marker='o', color='mediumpurple')
ax1.plot(USR, running_time_vbu, label='VBU (No Pri.)', linestyle='-', marker='D', color='darkgreen')
ax1.set_ylim(200, 300)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.legend()

# Lower plot (y-axis range: 0-50)
ax2.plot(USR, running_time_origin, linestyle='--', marker='s', color='lightgreen')
ax2.plot(USR, running_time_oul, linestyle='-', marker='o', color='mediumpurple')
ax2.plot(USR, running_time_vbu, linestyle='-', marker='D', color='darkgreen')
ax2.set_ylim(0, 50)
ax2.spines['top'].set_visible(False)

# Add diagonal lines to indicate a break in the y-axis
d = .015  # size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# Labels
ax2.set_xlabel('USR (%)')
ax2.set_ylabel('Running Time (s)')
fig.suptitle('Running Time vs. USR')

plt.show()
