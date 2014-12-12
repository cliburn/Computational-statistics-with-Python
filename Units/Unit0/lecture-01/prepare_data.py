import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    folder = 'lecture-01'
    n = 25
    p = 20000

    names = list(islice((''.join(perm) for perm in permutations('abcdefghi')), 0, p))
    cases = pd.DataFrame(np.random.normal(0, 1, (n, p)) , columns=names)
    ctrls = pd.DataFrame(np.random.normal(0, 1, (n, p)), columns=names)
    cases.to_csv(os.path.join(folder, 'cases.csv'), index=False)
    ctrls.to_csv(os.path.join(folder, 'ctrls.csv'), index=False)
    
