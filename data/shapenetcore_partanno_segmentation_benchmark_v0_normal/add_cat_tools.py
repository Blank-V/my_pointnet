import numpy as np
import os

txt_ls=os.listdir('./1')
path='./1'

for name in txt_ls:
    str = ''
    file_path=os.path.join(path,name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l=l.strip('\n')
            t = l.split(',')
            t = t[:6]
            t.append('1\n')
            t_str = ' '.join(t)
            str += "".join(t_str)

    with open(file_path, 'w') as f:
        f.write(str)

    print(name,'is done!')
print('end')