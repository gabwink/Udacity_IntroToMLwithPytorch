ps = 1/10000
pns = 1 - ps

acc = 99/100
nacc = 1 - acc

pp = ps * acc + pns * nacc

p_sp = (ps * acc)/pp
p_nsp = (pns * nacc)/pp

print(p_sp)
print(p_nsp)

print(120/12)