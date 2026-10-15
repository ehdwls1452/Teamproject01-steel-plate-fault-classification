import pandas as pd, numpy as np
def make_coil_groups(df, gap_thr=200000):
    """같은 Length_of_Conveyer + Y좌표가 연속으로 이어지는 덩어리 = 같은 코일로 간주"""
    d = df.reset_index().sort_values(['Length_of_Conveyer','Y_Minimum'])
    newg = (d.Length_of_Conveyer != d.Length_of_Conveyer.shift()) | \
           ((d.Y_Minimum - d.Y_Maximum.shift()) > gap_thr)
    d['grp'] = newg.cumsum()
    return d.sort_values('index')['grp'].values
