
lab = []
for i in range(6):
    for j in range(4):
        lab.extend(list([j+9]*9))
A = list([0]*9+[1]*9+[2]*9+[3]*9+[4]*9+[5]*6*9+[6]*6*9+[7]*5*9+[8]*3*9+lab+[13]*10*9+[14]*5*9)
print(A)