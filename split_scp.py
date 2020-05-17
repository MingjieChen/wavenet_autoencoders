


scp_path = './dump/2017/test/train.txt'

f = open(scp_path, 'r')
lines = f.readlines()

lan = 'LANG2'

new_f = open(scp_path + f'.{lan}', 'w', encoding = 'utf-8')

for l in lines:
    if lan in l:
        new_f.write(l)


