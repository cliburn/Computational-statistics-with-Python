import os
import numpy as np
import pandas as pd
import scipy.stats as st
from tabulate import tabulate
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    ntop = 30
    
    cases = pd.read_csv('cases.csv')
    ctrls = pd.read_csv('ctrls.csv')

    genes = ctrls.columns
    scores, pvals = st.ttest_ind(cases, ctrls)
    top_idx = np.argsort(pvals)[:ntop]
    df = pd.DataFrame(np.array([genes[top_idx], pvals[top_idx]]).T,
                      columns=['Gene', 'p'])

    with open('table.tex', 'w') as f:
        f.write(tabulate(df, headers=list(df.columns),
                         tablefmt="latex", floatfmt=".4f"))

    plt.figure(tight_layout=True)
    cmap = sns.diverging_palette(255, 1, n=3, as_cmap=True)
    sns.clustermap(pd.concat([ctrls.ix[:, top_idx].sort(axis=1),
                              cases.ix[:, top_idx].sort(axis=1)]).T,
                              cmap=cmap);
    plt.savefig('heatmap.png')
