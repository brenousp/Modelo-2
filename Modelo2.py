# -*- coding: utf-8 -*-

from gurobipy import *
import time
import numpy as np
import math

#Traduz para a nova API do Gurobi 13.0.1
# porque a versão dela é antiga:=====================================
# [compat] Gurobi 13: addConstr() no longer accepts (expr, GRB.OP, rhs).
# This shim restores the old 3-positional-argument behavior without touching
# any of Karasan's lines.
_OLD_ADD = Model.addConstr
def _addConstr_compat(self, expr, op=None, rhs=None, *args, **kwargs):
    if op is None:
        return _OLD_ADD(self, expr, *args, **kwargs)
    if op == GRB.EQUAL:
        return _OLD_ADD(self, expr == rhs, *args, **kwargs)
    if op == GRB.GREATER_EQUAL:
        return _OLD_ADD(self, expr >= rhs, *args, **kwargs)
    if op == GRB.LESS_EQUAL:
        return _OLD_ADD(self, expr <= rhs, *args, **kwargs)
    raise ValueError("Unsupported operator")
Model.addConstr = _addConstr_compat
#========================================================================

#%%######################################### Choose options ####################################
inFile = input("Please enter protein's name:")
nameofprotein = inFile
inFile = inFile + ".txt"
outFile = input("Please enter the output file name:")
outFile = outFile + ".txt"
print(" ")

print("Options for objective function:")
print("1: Maximize CPB")
print("2: Maximize CAI")
print("3: Minimize RCPB")
print("4: Minimize RCB")
objectivefunction = input("Please enter the objective function:")
print("")
objectivefunction = int(objectivefunction)

print("Options for the constraint:")

if objectivefunction != 1:
    CPBconst = input("Enter treshhold value for CPB (if you do not want this constraint to be activated type 'N'):")
if objectivefunction != 2:
    CAIconst = input("Enter treshhold value for CAI (if you do not want this constraint to be activated type 'N'):")
if objectivefunction != 3:
    RCPBconst= input("Enter treshhold value for RCPB (if you do not want this constraint to be activated type 'N'):")
if objectivefunction != 4:
    RCBconst= input("Enter treshhold value for RCB (if you do not want this constraint to be activated type 'N'):")


#%%######################################### Read data and generate parameters ####################################
inputFile=open(inFile,"r")
outputFile=open(outFile,"w+")

##Read data from text file
data=open(inFile).readlines()
for n,line in enumerate(data):
    data[n]=line.rstrip()
newdata=""
for n in range(len(data)):
    newdata=newdata+data[n]
newdata=newdata.replace(' ','')

array=list(newdata)

##20 aminoacids and one stop symbol
#aminoacidset=['Isoleucine', 'Leucine','Valine','Phenylalanine','Methionine','Cysteine','Alanine','Glycine','Proline','Threonine','Serine','Tyrosine','Tryptophan','Glutamine','Asparagine','Histidine','Glutamic acid','Aspartic acid','Lysine','Arginine','Stop codons'];
##20 aminoacids and one stop symbol
aminoacidset=['I','L','V','F','M','C','A','G','P','T','S','Y','W','Q','N','H','E','D','K','R','X']
##64 codons
codonset=['AUU','AUA','AUC','CUA','CUC','CUG','CUU','UUA','UUG','GUU','GUA','GUC','GUG','UUU','UUC','AUG','UGU','UGC','GCA','GCC','GCG','GCU','GGU','GGC','GGA','GGG','CCU','CCC','CCA','CCG','ACU','ACC','ACA','ACG','UCU','UCC','UCA','UCG','AGU','AGC','UAU','UAC','UGG','CAA','CAG','AAU','AAC','CAU','CAC','GAA','GAG','GAU','GAC','AAA','AAG','CGU','CGC','CGA','CGG','AGA','AGG','UAA','UAG','UGA'];

##Take aminoacid sequence of protein as input

N=len(array)
aminoacids=[0]*N
for i in range(N):
    aminoacids[i]=array[i]


##Y matrix whose entry yij  is equal to 1 if ith amino acid in the protein is the j th amino acid in our list.

Y= np.zeros((N,len(aminoacidset)),dtype=int)   # [compat] np.int -> int (NumPy >= 1.24)
for i in range(len(aminoacids)):
    for j in range(len(aminoacidset)):
        if aminoacids[i]==aminoacidset[j]:
            Y[i,j]=1


test2=np.sum(Y, axis = 1)

##M matrix whose entry mjk  is equal to 1 if jth amino acid can be represented by codon k.
if len(test2)!=N:
    print('warning! there exists undefined AA letter abbreviation')

##M matrix whose entry mjk  is equal to 1 if jth amino acid can be represented by codon k.

M= np.zeros((len(aminoacidset),len(codonset)),dtype=int)   # [compat] np.int -> int
M[0,0:3]=[1,1,1]
M[1,3:9]=[1,1,1,1,1,1]
M[2,9:13]=[1,1,1,1]
M[3,13:15]=[1,1]
M[4,15:16]=[1]
M[5,16:18]=[1,1]
M[6,18:22]=[1,1,1,1]
M[7,22:26]=[1,1,1,1]
M[8,26:30]=[1,1,1,1]
M[9,30:34]=[1,1,1,1]
M[10,34:40]=[1,1,1,1,1,1]
M[11,40:42]=[1,1]
M[12,42:43]=[1]
M[13,43:45]=[1,1]
M[14,45:47]=[1,1]
M[15,47:49]=[1,1]
M[16,49:51]=[1,1]
M[17,51:53]=[1,1]
M[18,53:55]=[1,1]
M[19,55:61]=[1,1,1,1,1,1]
M[20,61:64]=[1,1,1]


#R matrix for possible codonset of amino acids in the protein
R= np.zeros((len(aminoacids),len(codonset)),dtype=int)   # [compat] np.int -> int
for i in range(len(aminoacids)):
    for j in range(len(aminoacidset)):
        if aminoacids[i]==aminoacidset[j]:
            R[i]=M[j]
            break

##Codon pair bias
inFileCPB="cpb.txt"
CPB=open(inFileCPB).readlines()
for n,line in enumerate(CPB):
    CPB[n]=line.split("\t")

##Fitness Values
inFileFV="cai.txt"
with open(inFileFV) as f:
    FitnessValues = f.read().split(',')

logFitnessValues=np.zeros(64,dtype=np.double)
for i in range(len(FitnessValues)):
    logFitnessValues[i]=np.log(float(FitnessValues[i]))

## Codon Frequency
inFileFC="rcb.txt"
FreqCodons= np.zeros(64,dtype=np.double)
with open(inFileFC) as f:
    FreqCodons = f.read().split(',')

## Codon Pair Frequency
inFileFCP= "rcpb.txt"
FreqCodonPairs=open(inFileFCP).readlines()
for n,line in enumerate(FreqCodonPairs):
    FreqCodonPairs[n]=line.split("\t")

##Frequency of amino acid pair
NumAA= np.zeros(len(aminoacidset),dtype=int)   # [compat] np.int -> int
NumAA=np.sum(Y, axis = 0)

NumAApairs= np.zeros((len(aminoacidset),len(aminoacidset)),dtype=int)   # [compat] np.int -> int
for j in range(len(aminoacidset)):
    for k in range(len(aminoacidset)):
        for i in range(len(aminoacids)-1):
            if Y[i,j]==1 and Y[i+1,k]==1:
                NumAApairs[j,k]=NumAApairs[j,k]+1


eta = np.zeros(len(aminoacidset),dtype=int)   # [compat] np.int -> int
eta=np.sum(Y, axis = 0)


etapair= np.zeros((len(codonset),len(codonset)),dtype=int)   # [compat] np.int -> int
for j in range(len(codonset)):
    for k in range(len(codonset)):
        for i in range(len(aminoacidset)):
            for r in range(len(aminoacidset)):
             if M[i,j]==1 and M[r,k]==1:
                etapair[j,k]=NumAApairs[i,r]


NumAminoAcidCodonPossibility=np.zeros((len(aminoacidset)),dtype=int)   # [compat] np.int -> int
NumAminoAcidCodonPossibility=(np.sum(M, axis = 1))


NumAminoAcidPairCodoPairPossibility=np.zeros((len(aminoacidset),len(aminoacidset)),dtype=int)   # [compat] np.int -> int
for i in range(len(aminoacidset)):
    for r in range(len(aminoacidset)):
          NumAminoAcidPairCodoPairPossibility[i,r]=np.sum(M[i])*np.sum(M[r])


# =============================================================================
# ===== EXTENSÃO B.0: Minhas restrições/parâmetros  ===========================
# =============================================================================

MIN_STOPS_TOTAL        = 20
JANELA_STOP            = 30
MIN_STOPS_POR_JANELA   = 1
LAMBDA_SOFT            = 0.05
FILE_MOTIFS            = "Conjunto de restrições U.CSV"

# Motifs indesejados
def carregar_motifs(caminho):
    import csv
    with open(caminho, encoding="utf-8") as f:
        primeira = f.readline()
        sep = ";" if ";" in primeira else ","
    out = []
    with open(caminho, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=sep):
            m = (row.get("motif","") or "").strip().upper()
            if not m: continue
            p = float(row.get("peso","0") or 0)
            if p > 0:
                out.append({"motif": m, "peso": p})
    return out

# Pesos dos motifs
motifs_todos = carregar_motifs(FILE_MOTIFS)
motifs_hard  = [m for m in motifs_todos if m["peso"] >= 0.7]
motifs_soft  = [m for m in motifs_todos if m["peso"] <  0.7]
print(f"Motifs hard: {len(motifs_hard)} | soft: {len(motifs_soft)}")

def precomputar_motifs(motifs):
    comp = []
    for mu_dict in motifs:
        mu, peso, L = mu_dict["motif"], mu_dict["peso"], len(mu_dict["motif"])
        for i in range(1, 3*N - L + 2):
            s, t = math.ceil(i/3), math.ceil((i+L-1)/3)
            aminoacidos_c, viavel = [], True
            for k in range(s, t+1):
                ini = max(3*(k-1)+1, i)
                fim = min(3*k, i+L-1)
                if ini > fim: continue
                seg = mu[ini-i:fim-i+1]
                pos = ini - (3*(k-1)+1)
                cc = [idx for idx in range(64)
                      if R[k-1,idx]==1
                      and codonset[idx][pos:pos+len(seg)]==seg]
                if not cc:
                    viavel = False; break
                aminoacidos_c.append((k, cc))
            if viavel and aminoacidos_c:
                comp.append({"i":i,"mu":mu,"peso":peso,"aminoacidos":aminoacidos_c})
    return comp

compativeis_hard = precomputar_motifs(motifs_hard)
compativeis_soft = precomputar_motifs(motifs_soft)
print(f"Restrições de Motifs Hard pré-computadas: {len(compativeis_hard)} | soft: {len(compativeis_soft)}")

# Codons de parada no quadro -1 
STOP_SUFIXOS = {"AA","AG","GA"}
combinacoes_stop = {}
for i in range(N-1):
    for k in range(64):
        if R[i,k]!=1 or codonset[k][2]!="U": continue
        for l in range(64):
            if R[i+1,l]==1 and codonset[l][:2] in STOP_SUFIXOS:
                combinacoes_stop[(i,k,l)] = "U"+codonset[l][:2]
print(f"Combinações que geram Stop no quadro -1: {len(combinacoes_stop)}")

#===============================================================================================
# O Modelo

m=Model("Codon Optimization")
m.setParam('TimeLimit', 3600)

##Variable Zijk=1 if codon pair jk is used for amino acids i and i+1
coef={}
Z={}
for i in range(len(aminoacids)-1):
    for j in range(len(codonset)):
        for k in range(len(codonset)):
            if (R[i,j]==1 & R[i+1,k]==1):
                Z[i,j,k]=m.addVar(vtype=GRB.BINARY, name="Z%s" % str([i,j,k]))
                coef[i,j,k]=CPB[j][k]
m.update()

## Flow balance equations
for i in range(len(aminoacids)):
    jvar=[]
    kvar=[]
    hvar=[]
    for j in range(len(codonset)):
        if R[i,j]==1:
            jvar.append(j)
    for k in range(len(codonset)):
        if i != len(aminoacids)-1 and R[i+1,k]==1:
            kvar.append(k)
    for h in range(len(codonset)):
        if i !=0 and R[i-1,h]==1:
            hvar.append(h)
    if i == 0:
        m.addConstr(sum(Z[i,l,n] for l in jvar for n in kvar), GRB.EQUAL, 1)
    if i != 0 and i!= len(aminoacids)-1:
        for l in jvar:
            m.addConstr((sum(Z[i,l,n] for n in kvar) - sum(Z[i-1,h,l] for h in hvar)), GRB.EQUAL, 0)


## Second Problem Constraints
if objectivefunction == 3 or objectivefunction == 4 or RCPBconst != "N" or RCBconst != "N":
    codondevpositive={}
    for i in range(len(codonset)):
        codondevpositive[i]=m.addVar(vtype=GRB.CONTINUOUS, name="codondevpositive%s" % str([i]))
    m.update()

    for (i) in codondevpositive:
        m.addConstr(codondevpositive[i], GRB.GREATER_EQUAL, 0)

    codondevnegative={}
    for i in range(len(codonset)):
        codondevnegative[i]=m.addVar(vtype=GRB.CONTINUOUS, name="codondevnegative%s" % str([i]))
    m.update()

    for (i) in codondevnegative:
        m.addConstr(codondevnegative[i], GRB.GREATER_EQUAL, 0)

    for j in range(len(codonset)):
        xvar=[]
        etatemp=[]
        for k in range(len(aminoacidset)):
            if M[k,j]==1:
                etatemp=eta[k]
        for i in range(len(aminoacids)):
            if R[i,j]==1 :
                xvar.append(i)
        codonsum = 0
        if etatemp!=0:
            for i in xvar:
                for h in range(len(codonset)):
                    if i != N-1 and R[i+1,h] == 1:
                        codonsum += Z[i,j,h]
                    if i == N-1 and R[i-1,h] == 1:
                       codonsum += Z[i-1,h,j]
            m.addConstr(100*codonsum/etatemp, GRB.EQUAL,100*(float(FreqCodons[j])+codondevpositive[j]-codondevnegative[j]))
            m.update()
    m.update()


    codonpairdevpos={}
    for j in range(len(codonset)):
        for k in range(len(codonset)):
            codonpairdevpos[j,k]=m.addVar(vtype=GRB.CONTINUOUS, name="codonpairdevpos%s" % str([j,k]))
    m.update()



    codonpairdevneg={}
    for j in range(len(codonset)):
        for k in range(len(codonset)):
            codonpairdevneg[j,k]=m.addVar(vtype=GRB.CONTINUOUS, name="codonpairdevneg%s" % str([j,k]))
    m.update()

    for (i,j) in codonpairdevpos:
        m.addConstr(codonpairdevpos[i,j], GRB.GREATER_EQUAL, 0)

    m.update()

    for (i,j) in codonpairdevneg:
        m.addConstr(codonpairdevpos[i,j], GRB.GREATER_EQUAL, 0)

    m.update()


    for j in range(len(codonset)):
        for k in range(len(codonset)):
            ivar=[]
            etapairtemp=[]

            for i in range(len(aminoacids)-1):
                if R[i,j]==1 and R[i+1,k]==1:
                    ivar.append(i)

            if len(ivar)!= 0 :
                etapairtemp=len(ivar)
                m.addConstr(100*sum(Z[i,j,k]for i in ivar)/etapairtemp , GRB.EQUAL,100*(float(FreqCodonPairs[j][k])+ codonpairdevpos[j,k]-codonpairdevneg[j,k]))
    m.update()

    AAdev={}
    for i in range(len(aminoacidset)):
        AAdev[i]=m.addVar(vtype=GRB.CONTINUOUS, name="AAdev%s" % str([i]))

    m.update()

    for i in range(len(aminoacidset)):
        jvar=[]
        for j in range(len(codonset)):
            if M[i,j]==1:
              jvar.append(j)
        m.addConstr(100*sum((codondevpositive[j] + codondevnegative[j]) for j in jvar)/ NumAminoAcidCodonPossibility[i] , GRB.EQUAL, 100*AAdev[i])
    m.update()


    AApairdev={}
    for i in range(len(aminoacidset)):
        for j in range(len(aminoacidset)):
            AApairdev[i,j]=m.addVar(vtype=GRB.CONTINUOUS, name="AApairdev%s" % str([i,j]))


    for i in range(len(aminoacidset)):
        for j in range(len(aminoacidset)):
            kvar=[]
            lvar=[]
            for k in range(len(codonset)):
                if  M[i,k]==1:
                    kvar.append(k)
            for l in range(len(codonset)):
                if  M[j,l]==1:
                    lvar.append(l)
            m.addConstr(100*sum((codonpairdevpos[k,l]+codonpairdevneg[k,l]) for k in kvar for l in lvar)/NumAminoAcidPairCodoPairPossibility[i,j] , GRB.EQUAL, 100*(AApairdev[i,j]))

m.update()


if objectivefunction != 1:
    if CPBconst != "N":
        minCPB = float(CPBconst)
        m.addConstr(sum(Z[i,j,k]*coef[i,j,k] for (i,j,k) in Z)/(N-1), GRB.GREATER_EQUAL, minCPB)
        m.update()

if objectivefunction != 2:
    if CAIconst != "N":
        minCAI =N*math.log(float(CAIconst))
        m.addConstr(sum(Z[i,j,k]*logFitnessValues[j] for (i,j,k) in Z) + sum(Z[i,j,k]*logFitnessValues[k] for (i,j,k) in Z if i == N-2), GRB.GREATER_EQUAL, minCAI)
        m.update()

if objectivefunction != 3:
    if RCPBconst != "N":
        maxRCPB=float(RCPBconst)*N
        m.addConstr(100*sum(AApairdev[i,j]*NumAApairs[i,j] for (i,j) in AApairdev) , GRB.LESS_EQUAL, 100*maxRCPB)
        m.update()

if objectivefunction != 4:
    if RCBconst != "N":
        maxRCB=float(RCBconst)*N
        m.addConstr(100*sum(AAdev[j]*NumAA[j] for (j) in AAdev) , GRB.LESS_EQUAL, 100*maxRCB)
        m.update()



# =============================================================================
# ===== Adaptando minhas variáveis x para X ===================================
# =============================================================================
# A formulção em newtwork da Karasan não tem x. Para reutilizar as restrições 
# eu recontruí em "flow Z". Desde que o Z seja binário, o x vai ser 0/1 automaticamente.
x = {}
for i in range(len(aminoacids)):
    for k in range(len(codonset)):
        if R[i,k] == 1:
            x[i,k] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x%s" % str([i,k]))
m.update()

for i in range(len(aminoacids)):
    for k in range(len(codonset)):
        if R[i,k] != 1:
            continue
        if i == 0:
            kvar = [kk for kk in range(len(codonset)) if R[1,kk]==1]
            m.addConstr(x[0,k] == sum(Z[0,k,kk] for kk in kvar))
        elif i == len(aminoacids)-1:
            hvar = [h for h in range(len(codonset)) if R[len(aminoacids)-2,h]==1]
            m.addConstr(x[i,k] == sum(Z[len(aminoacids)-2,h,k] for h in hvar))
        else:
            hvar = [h for h in range(len(codonset)) if R[i-1,h]==1]
            m.addConstr(x[i,k] == sum(Z[i-1,h,k] for h in hvar))
m.update()

# =============================================================================
# ===== Restrição de motifs hard ==============================================
# =============================================================================
for idx, c in enumerate(compativeis_hard):
    termos = [x[k-1, idx_codon] for (k, cc) in c["aminoacidos"] for idx_codon in cc]
    if termos:
        m.addConstr(sum(termos) <= len(c["aminoacidos"]) - 1, name="hard_%d" % idx)

# =============================================================================
# ===== Restrições de motifs soft) ============================================
# =============================================================================
y_soft = {}
for idx, c in enumerate(compativeis_soft):
    y_soft[idx] = m.addVar(vtype=GRB.BINARY, name="ysoft_%d" % idx)
m.update()

for idx, c in enumerate(compativeis_soft):
    termos = [x[k-1, idx_codon] for (k, cc) in c["aminoacidos"] for idx_codon in cc]
    if termos:
        m.addConstr(-y_soft[idx] + sum(termos) <= len(c["aminoacidos"]) - 1,
                    name="soft_%d" % idx)

obj_penalidade = LAMBDA_SOFT * sum(
    c["peso"] * y_soft[idx] for idx, c in enumerate(compativeis_soft))

# =============================================================================
# ===== Restrição de stops no quadro -1 =======================================
# =============================================================================
w = {}
for (i,k,l) in combinacoes_stop.keys():
    w[i,k,l] = m.addVar(vtype=GRB.BINARY, name="w_%d_%d_%d" % (i,k,l)) #aproveitando as combinações
m.update()
for (i,k,l) in combinacoes_stop.keys():
    m.addConstr(w[i,k,l] <= Z[i,k,l])

s = {}
for i in range(len(aminoacids)-1):
    s[i] = m.addVar(vtype=GRB.BINARY, name="s_%d" % i)
m.update()

for i in range(len(aminoacids)-1):
    combos_i = [(k,l) for (ii,k,l) in combinacoes_stop.keys() if ii==i]
    if not combos_i:
        m.addConstr(s[i] == 0)
    else:
        for (k,l) in combos_i:
            m.addConstr(s[i] >= w[i,k,l])
        m.addConstr(s[i] <= sum(w[i,k,l] for (k,l) in combos_i))

m.addConstr(sum(s[i] for i in range(len(aminoacids)-1)) >= MIN_STOPS_TOTAL,
            name="Min_Stops_Total")
for i0 in range(len(aminoacids)-1 - JANELA_STOP + 1):
    m.addConstr(
        sum(s[i] for i in range(i0, i0+JANELA_STOP)) >= MIN_STOPS_POR_JANELA,
        name="janela_%d" % i0)


# =============================================================================
# ===== Função Obejetivo ==============
# =============================================================================

if objectivefunction == 1 :
    obj = sum(Z[i,j,k]*coef[i,j,k] for (i,j,k) in Z)/(N-1)
if objectivefunction == 2 :
    obj = sum(Z[i,j,k]*logFitnessValues[j] for (i,j,k) in Z) + sum(Z[i,j,k]*logFitnessValues[k] for (i,j,k) in Z if i == N-2)
if objectivefunction == 3 :
    obj = 100*sum(AApairdev[i,j]*NumAApairs[i,j] for (i,j) in AApairdev)
if objectivefunction == 4 :
    obj = sum(AAdev[j]*NumAA[j] for (j) in AAdev)/N

# Função de penalidade dos motifs soft
obj = obj - obj_penalidade

if objectivefunction == 1 or objectivefunction == 2:
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

if objectivefunction == 3 or objectivefunction == 4:
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()


# =============================================================================
# ===== Me mostra, em caso de infactiblidade, se eu exportei algo errado ======
# =============================================================================
if m.status == GRB.INFEASIBLE:
    print("\n=== MODEL IS INFEASIBLE ===")
    print("Computing IIS (Irreducible Inconsistent Subsystem)...")
    m.computeIIS()
    iis_file = outFile.replace(".txt", "_IIS.ilp")
    m.write(iis_file)
    print("IIS written to:", iis_file)
    print("Open it to see exactly which constraints conflict.")


if m.status == GRB.Status.OPTIMAL:
    print('\nobjective function value: %g' %  m.objVal)
    objValue=m.objVal

    for v in m.getVars():
        if v.x >=1:
            print('%s %g' % (v.varName, v.x))
#=======================================================================================

    #Write codons into the file
    codons = [(j) for (i,j,k) in Z if Z[i,j,k].X >= 0.9]
    codons = codons + [(k) for (i,j,k) in Z if Z[i,j,k].X >= 0.9 and i == N - 2]
    for j in range(len(codons)):
        outputFile.write(codonset[codons[j]])

    line1 = "\nName of the protein: " + nameofprotein
    outputFile.write(line1)
    outputFile.write('\nNumber of amino acids in the protein: %g' % N )

    if objectivefunction == 1 :
        outputFile.write('\nObjective is maximizing CPB')
    if objectivefunction == 2 :
        outputFile.write('\nObjective is maximizing CAI')
    if objectivefunction == 3 :
        outputFile.write('\nObjective is minimizing RCPB')
    if objectivefunction == 4 :
        outputFile.write('\nObjective is minimizing RCB')

    if objectivefunction != 1:
        if CPBconst != "N":
            outputFile.write('\nCPB constraint with treshhold value of  %g' % float(CPBconst))
    if objectivefunction != 2:
        if CAIconst != "N":
            outputFile.write('\nCAI constraint with treshhold value of %g' % float(CAIconst))
    if objectivefunction != 3:
        if RCPBconst != "N":
            outputFile.write('\nRCPB constraint with treshhold value of %g' % float(RCPBconst))
    if objectivefunction != 4:
        if RCBconst != "N":
            outputFile.write('\nRCB constraint with treshhold value of %g' % float(RCBconst))

    outputFile.write('\nObjective function value: %g' % objValue )
    runtime = m.Runtime
    outputFile.write('\nRun time is: %g' % runtime )

    print(objValue)

outputFile.close()
print("END")