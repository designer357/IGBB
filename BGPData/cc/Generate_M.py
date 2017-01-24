import os
outputfile = "HB_C_N_S_Multi.txt"
file1 = "HB_Code_Red_I.txt"
file2 = "HB_Nimda.txt"
file3 = "HB_Slammer.txt"

file_type_0 = "1\n"
file_type_1 = "2\n"
file_type_2 = "3\n"
file_type_3 = "4\n"
val_out = []

with open(os.path.join(os.getcwd(),file1))as fin:
    base = len(val_out)
    val1 = fin.readlines()
    for tab in range(len(val1)):
        val_out.append([])
        temp  = val1[tab].split(',')
        if temp[-1].strip() == '1.0\n'or temp[-1].strip() == '1.0':
            temp[-1] = file_type_0
        else:
            temp[-1] = file_type_1
        val_out[base+tab].extend(temp)
print(val1[0].split(',')[-1].strip())
with open(os.path.join(os.getcwd(),file2))as fin:
    base = len(val_out)
    val2 = fin.readlines()
    for tab in range(len(val2)):
        val_out.append([])
        temp  = val2[tab].split(',')
        if temp[-1].strip() == '1.0\n'or temp[-1].strip() == '1.0':
            temp[-1] = file_type_0
        else:
            temp[-1] = file_type_2
        val_out[base+tab].extend(temp)
print(len(val_out[0]))
with open(os.path.join(os.getcwd(),file3))as fin:
    base = len(val_out)
    val3 = fin.readlines()
    for tab in range(len(val3)):
        val_out.append([])
        temp  = val3[tab].split(',')
        if temp[-1].strip() == '1.0\n'or temp[-1].strip() == '1.0':
            temp[-1] = file_type_0
        else:
            temp[-1] = file_type_3
        val_out[base+tab].extend(temp)
print(val_out[0])
with open(os.path.join(os.getcwd(),outputfile),"w")as fout:
    for tab1 in range(len(val_out)):
        for tab2 in range(len(val_out[0])):
            if not '\n' in val_out[tab1][tab2]:
                fout.write(val_out[tab1][tab2]+',')
            else:
                fout.write(val_out[tab1][tab2])


