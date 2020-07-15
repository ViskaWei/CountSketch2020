truthV = l128 + 0.5 * l64
markerV = predimg==6
bkgc = [0.95,0.95,0.95,1]
l64c = [0.7,0.7,0.7,1]
l128c = [0.4,0.4,0.4,1]
markerc = 'red'

# from matplotlib. import colors

truthcmap, truthnorm = from_levels_and_colors([-0.1,0.3,0.6,1.1], [bkgc,l64c,l128c])
plt.figure(figsize=(20,16))
plt.imshow(truthV, cmap=truthcmap, norm=truthnorm)

plt.contour(markerV, colors=markerc, linewidths=0.5, alpha=1)
#plt.imshow(truthV[200:600,200:800], cmap=truthcmap, norm=truthnorm)
#plt.contour(markerV[200:600,200:800], colors='red', linewidths=0.8, alpha=1)

legend_elements = [Patch(facecolor=l128c, label='TN'),
                  Patch(facecolor=l64c, label='NTN'),
                  Patch(facecolor=[0,0,0,0], edgecolor=markerc, label=f'Pred_{ii}')]
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
plt.legend(handles=legend_elements, loc='upper left', fontsize='xx-large')
# plt.savefig('forlordviska.png')

s=np.loadtxt(f'{PRETRAIN}/eval.txt')
pc=np.loadtxt(f'{PRETRAIN}/pc.txt')