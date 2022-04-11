import os

txt_ls=os.listdir('./1')
"shape_data/03001627/355fa0f35b61fdd7aa74a6b5ee13e775"
res=[]
for name in txt_ls:
    name_pure= name[:-4]
    path="\"shape_data/1/"+name_pure+"\""
    res.append(path)
    res.append(",")

str= ''.join(res)
with open('./path_list.txt','w')as f:
    f.write(str)
print(res)