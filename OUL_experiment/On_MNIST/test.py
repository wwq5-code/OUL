import matplotlib.pyplot as plt

# Define the words and their properties (size, position, rotation, color)
words = [
    {'text': 'Confidentiality', 'size': 50, 'position': (0.5, 0.8), 'rotation': 0, 'color': 'red'},
    {'text': 'Privacy', 'size': 45, 'position': (0.2, 0.7), 'rotation': 90, 'color': 'green'},
    {'text': 'Security', 'size': 40, 'position': (0.5, 0.55), 'rotation': -0, 'color': 'blue'},
    {'text': 'Autonomy', 'size': 35, 'position': (0.3, 0.5), 'rotation': -90, 'color': 'purple'},
    {'text': 'Consent', 'size': 30, 'position': (0.6, 0.4), 'rotation': -90, 'color': 'orange'},
    {'text': 'Trust', 'size': 25, 'position': (0.4, 0.3), 'rotation': 0, 'color': 'brown'},
    {'text': 'NonMaleficence', 'size': 20, 'position': (0.7, 0.2), 'rotation': 0, 'color': 'pink'},
    {'text': 'Beneficence', 'size': 20, 'position': (0.2, 0.2), 'rotation': -0, 'color': 'gray'}
]

# Create a plot
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Add words to the plot
for word in words:
    plt.text(word['position'][0], word['position'][1], word['text'],
             fontsize=word['size'], rotation=word['rotation'], color=word['color'],
             ha='center', va='center')

# Hide axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Show plot
plt.show()
