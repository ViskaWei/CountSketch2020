###############plotting###################
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
rcParams['figure.figsize']=(16,10)
##########################################
ckeys=['ug','ur', 'ui','uz','gz', 'gi', 'gr','rz','ri','iz']
mkeys=['z']
ftr=ckeys+mkeys
luclr_dict = dict({'I':'#0D2B68',
                  'II':'maroon',
                  'III': 'red',
                  'IV': '#FF5003',
                   'V': '#FACF4A',
                  'W':'k'})
                  
def plot_spec_HH(HH_pdQS,base,ftr=ftr,hues=['classU','l5U','l8U']):
    f,axs=plt.subplots(1,3,figsize=(20,8))
    for ii, ax in enumerate(axs.flatten()):
        sns.scatterplot('u1','u2',data=HH_pdQS,hue=hues[ii],ax=ax)
        if ii==0: ax.set_title(f'base{base}', fontsize=16)
        if ii==1: ax.set_title(f'{ftr}', fontsize=16)
    return None

# f, (ax0,ax1)=plt.subplots(1,2, figsize=(16,10))
# sns.scatterplot(x='u1',y='u2',data=photoUTe, s=5, color='gray',marker="+", alpha=0.2,ax=ax0, label='Photo HHs')
# sns.scatterplot(x='u1',y='u2',data=photoUTe, s=5, color='gray',marker="+", alpha=0.2,ax=ax1)
# sns.scatterplot('u1','u2',data=HH_pdQS,hue='l8U', s=10,marker='x',ax=ax1 )plt.figure(figsize=(16,10))
# g = sns.FacetGrid(HH_pdQS, hue="l8U",size=5)
# g.map(sns.kdeplot, "u1", "u2", alpha=1, n_levels=3)
# g.add_legend();