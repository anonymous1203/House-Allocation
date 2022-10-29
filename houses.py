import gurobipy as gp
import numpy as np
from gurobipy import Model, GRB, quicksum 
import itertools 
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

import logging
logging.basicConfig(filename="log.txt", level=logging.DEBUG, filemode = "w")


Agents_Houses_Types = [[15, 15, 1], [15,15,3], [15, 15, 5], [15, 15, 15],[15, 20, 1], [15, 20, 3], [15, 20, 5],[30, 30, 1], [30, 30, 3], [30, 30, 5], [30, 30, 15], [30, 30, 30], [30, 40, 1], [30, 40,3], [60, 60, 1], [60, 60, 3], [60, 60, 5],[60, 60, 15],[60, 60, 30],
[90, 90, 1], [90, 90, 3], [90, 90, 5], [90, 90, 15], [90, 90, 30],[90, 100, 1], [90, 100, 3], [90, 100, 5], [120, 120, 1], [120, 120, 3], [120,120, 5], [120, 120, 5], [120, 120, 15], [120, 120, 30],[120, 130, 1], [120, 130,3], [120, 130, 5]]

# Agents_Houses_Types = [[15, 15, 1]]

table = []
for num in range(len(Agents_Houses_Types)):
    # M will store 100 valuation matrices, and the following lists store the 100 entries corresponding to the solutions of each matix A in M.
    # Finally the average is taken over these 100 solutions
    M=[]
    EnviousAgents = []
    EnvyAmount = [] 
    MaxEnvy=[]


    EgalitarianEnvy = []
    EnviousAgentsinEHA = []
    EnvyinEHA = []


    UtilitarianEnvy =[]
    EnviousAgentsinUHA =[]
    MaxEnvyinUHA = []


    DummyHouses = []
    MaxAgentDegree=[]
    MaxHouseDegree=[]
    Sparsity = []
   
    for i in range(1):
        start = time.time()
        # -------------------------------------
        # ----------Random matrices -----------

        A = np.random.randint(2, size=(Agents_Houses_Types[num][2],Agents_Houses_Types[num][1])) 
        A = np.repeat(A, Agents_Houses_Types[num][0]/Agents_Houses_Types[num][2], axis=0) 

        # -------------------------------------
        # --------Consecutive Ones Matrices---

        # A= np.zeros((Agents_Houses_Types[num][2],Agents_Houses_Types[num][1]), dtype=np.int16)
        # for row in range(len(A)):
        #     startingones = np.random.randint(0, Agents_Houses_Types[num][1]-1)
        #     endingones = np.random.randint(startingones,Agents_Houses_Types[num][1])
        #     for column in range(len(A[0])):
        #         if column >= startingones and column <= endingones:
        #             A[row][column]=1
        # A = np.repeat(A, Agents_Houses_Types[num][0]/Agents_Houses_Types[num][2], axis=0) 


        M.append(A) 
    print(M[0])


    # nstar stores the agent types
    for A in M:
        start = time.time() 
        nstar = []
        for i in range(len(A)):  
            counter=1
            for rows in nstar:
                if i in rows:
                    counter = 0
            if counter:  
                q=[i] 
                for j in range(i + 1, len(A)):   
                    if np.array_equal(A[i], A[j]): 
                        q.append(j)
                        
                nstar.append(q)
            else:
                continue
        print("Agent types", nstar)


        # mstar stores the house types
        mstar = []
        for i in range(len(A[0])):  
            counter=1
            for rows in mstar:
                if i in rows:
                    counter = 0
            if counter:  
                q=[i] 
                for j in range(i + 1, len(A[0])):   
                    if np.array_equal(A[:,i], A[:,j]): 
                        q.append(j) 
                mstar.append(q)
            else:
                continue
        print("House types", mstar)

        MaxAgentDegree.append(max(np.count_nonzero(A, axis=1)))
        MaxHouseDegree.append(max(np.count_nonzero(A, axis=0)))
        Sparsity.append(np.count_nonzero(A) / (len(A) * len(A[0])))

        dummy=0
        for h in range(len(A[0])):
            flag=0
            for a in range(len(A)):
                if A[a][h] > 0:
                    flag = 1
            if flag==0:
                dummy +=1
        DummyHouses.append(dummy)

        D={} # Dictionary D: key i is the agent type i and D[i] stores the house types that agent i likes.

        for i in range(len(nstar)):
            for j in range(len(mstar)):
                if A[nstar[i][0]][mstar[j][0]] == 1:
                    if i in D.keys():
                        D[i].append(j)
                    else:
                        D[i] = [j]

        # print(D)

        m = Model()

        #Adding variables
        z = m.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="z" )
        x = m.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="x")
        xhat = m.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="xhat")
        d= m.addVars(len(nstar), len(mstar), vtype=GRB.BINARY, name="d")
        dhat= m.addVars(len(nstar), len(mstar), vtype=GRB.BINARY, name="dhat")

        m.update()

        m.setObjective(quicksum(z[i, j] for i in range(len(nstar)) for j in range(len(mstar))), GRB.MINIMIZE)

        # print("length of nstar = number of agent types:", len(nstar))
        # print("length of mstar = number of house types:", len(mstar))

        #constraint 1: the number of houses allocated to n_i agents of type i is exactly n_i.
        for i in range(len(nstar)):
            m.addConstr(quicksum(x[i,j] for j in range(len(mstar))) == len(nstar[i]))

        #constraint 2: the number of allocated houses of type j does not exceed the available houses of type j, that is, m_j
        for j in range(len(mstar)):
            m.addConstr(quicksum(x[i,j] for i in range(len(nstar))) <= len(mstar[j]))

        #constraint 3: 
        for i in range(len(nstar)):
            for j in range(len(mstar)): 
                if j not in D[i]:
                    m.addConstr(x[i,j] <= len(A) * dhat[i, j])
                    m.addConstr(quicksum(x[k, l] for k in range(len(nstar)) for l in D[i]) <= len(A) * len(A[0]) * z[i, j] + len(A) * len(A[0]) * (1-dhat[i, j]))
                    m.addConstr(z[i, j] <= len(nstar[i]) * (quicksum(x[k, l] for k in range(len(nstar)) for l in D[i])))


        # constraint 4:
        for i in range(len(nstar)):
            for j in range(len(mstar)):
                m.addConstr(z[i,j] <= len(nstar[i]) * d[i, j])
                m.addConstr(x[i, j]-z[i,j] <= len(nstar[i]) * (1-d[i, j]))
                m.addConstr(z[i, j]<=x[i,j])



        # constraint 5: an agent who gets what he likes is not envious
        for i in range(len(nstar)):
            for j in D[i]:
                m.addConstr(z[i,j] == 0)



        # #constraint 6: 
        for i in range(len(nstar)):
            for j in range(len(mstar)):
                m.addConstr(x[i, j]>=0)
                m.addConstr(z[i, j]>=0)
                m.addConstr(d[i, j]>=0)
                m.addConstr(d[i, j]<=1)
                m.addConstr(dhat[i, j]>=0)
                m.addConstr(dhat[i, j]<=1)


        m.optimize()

        # for i in range(len(nstar)):
        #     for j in range(len(mstar)): 
        #         if j not in D[i]:
        #             print(x[i,j])

        EnviousAgents.append(m.objVal)
        # print(f"Optimal objective value: {m.objVal}")
        totalenvy = 0 
        maxenvy = 0
        for i in range(len(nstar)):
            envy=0
            for j in range(len(mstar)): 
                #if i is allocated a house j not in his approved set D[i]
                if j not in D[i] and x[i,j].X > 0:  
                    #then, check if houses in D[i] are allocated (the number of allocated houses from D[i] is the exact amount of envy experienced by agent i.)
                    for k in D[i]:
                        for agents in range(len(nstar)):
                            totalenvy = totalenvy + x[i, j].X * x[agents, k].X
                            envy = envy + x[agents, k].X
                            if envy > maxenvy:
                                maxenvy = envy


        EnvyAmount.append(totalenvy)
        MaxEnvy.append(maxenvy)

        OHAtime = time.time()-start

        #----------------------------------------------
        #---------------- EHA Model -------------------
        start1=time.time()
        ml = Model()

        #Adding variables
        Z = ml.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="Z" )
        X = ml.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="X")
        Xhat = ml.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="Xhat")
        Dd = ml.addVars(len(nstar), len(mstar), vtype=GRB.BINARY, name="Dd")
        Ddhat= ml.addVars(len(nstar), len(mstar), vtype=GRB.BINARY, name="Ddhat")
        w=ml.addVar(vtype= GRB.INTEGER, name = "w")

        ml.update()

        ml.setObjective(w, GRB.MINIMIZE)


        #constraint 1: the number of houses allocated to n_i agents of type i is exactly n_i.
        for i in range(len(nstar)):
            ml.addConstr(quicksum(X[i,j] for j in range(len(mstar))) == len(nstar[i]))

        #constraint 2: the number of allocated houses of type j does not exceed the available houses of type j, that is, m_j
        for j in range(len(mstar)):
            ml.addConstr(quicksum(X[i,j] for i in range(len(nstar))) <= len(mstar[j]))

        #constraint 3: 
        for i in range(len(nstar)):
            for j in range(len(mstar)): 
                if j not in D[i]:
                    ml.addConstr(X[i,j] <= len(A) * Ddhat[i, j])
                    ml.addConstr(quicksum(X[k, l] for k in range(len(nstar)) for l in D[i]) <= len(A) * len(A[0]) * Z[i, j] + len(A) * len(A[0]) * (1-Ddhat[i, j]))


        # constraint 4:
        for i in range(len(nstar)):
            for j in range(len(mstar)):
                ml.addConstr(Z[i,j] <= len(A) * Dd[i, j])
                ml.addConstr(quicksum(X[k, l] for k in range(len(nstar)) for l in D[i]) - Z[i,j] <= len(A) * (1-Dd[i, j]))
                ml.addConstr(Z[i, j]<= len(A) * quicksum(X[k, l] for k in range(len(nstar)) for l in D[i]))


        # constraint 5: an agent who gets what he likes is not envious
        for i in range(len(nstar)):
            for j in D[i]:
                ml.addConstr(Z[i,j] == 0)

        ml.addConstr(w>=0)

        # #constraint 6: 
        for i in range(len(nstar)):
            for j in range(len(mstar)):
                ml.addConstr(X[i, j]>=0)
                ml.addConstr(Z[i, j]>=0)
                ml.addConstr(Dd[i, j]>=0)
                ml.addConstr(Dd[i, j]<=1)
                ml.addConstr(Ddhat[i, j]>=0)
                ml.addConstr(Ddhat[i, j]<=1)
                ml.addConstr(Z[i,j]<=w)



        ml.optimize()
        # print(f"Optimal maxenvy value: {ml.objVal}")
        EgalitarianEnvy.append(ml.objVal)


        count= 0
        totalenvyinEHA = 0
        check=0
        for i in range(len(nstar)):
            for j in range(len(mstar)): 
                if j not in D[i] and X[i,j].X > 0:  
                    for k in D[i]:
                        for agents in range(len(nstar)):
                            if X[agents, k].X > 0:
                                count += X[i, j].X
                                totalenvyinEHA += X[i, j].X * X[agents, k].X


            
        EnviousAgentsinEHA.append(count)
        EnvyinEHA.append(totalenvyinEHA)

        # for v in ml.getVars():
        #     print('%s %g' % (v.varName, v.x))
        # print(check)
        # print("-------------")

        # for v in m.getVars():
        #     print('%s %g' % (v.varName, v.x))


        EHAtime = time.time()-start1
        start2 = time.time()


        #----------------------------------------------
        #---------------UHA Model----------------------


        # m_util = Model("qp")

        # #Adding variables
        # Z_util = m_util.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="Z_util" )
        # X_util = m_util.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="X_util")
        # Xhat_util = m_util.addVars(len(nstar), len(mstar), vtype=GRB.INTEGER, name="Xhat_util")
        # D_util = m_util.addVars(len(nstar), len(mstar), vtype=GRB.BINARY, name="D_util")
        # Dhat_util= m_util.addVars(len(nstar), len(mstar), vtype=GRB.BINARY, name="Dhat_util")
    

        # m_util.update()

        # m_util.setObjective(quicksum(X_util[i,j]*Z_util[i,j] for i in range(len(nstar)) for j in range(len(mstar))), GRB.MINIMIZE)

        # #constraint 1: the number of houses allocated to n_i agents of type i is exactly n_i.
        # for i in range(len(nstar)):
        #     m_util.addConstr(quicksum(X_util[i,j] for j in range(len(mstar))) == len(nstar[i]))

        # #constraint 2: the number of allocated houses of type j does not exceed the available houses of type j, that is, m_j
        # for j in range(len(mstar)):
        #     m_util.addConstr(quicksum(X_util[i,j] for i in range(len(nstar))) <= len(mstar[j]))

        # #constraint 3: 
        # for i in range(len(nstar)):
        #     for j in range(len(mstar)): 
        #         if j not in D[i]:
        #             m_util.addConstr(X_util[i,j] <= len(A) * Dhat_util[i, j])
        #             m_util.addConstr(quicksum(X_util[k, l] for k in range(len(nstar)) for l in D[i]) <= len(A) * len(A[0]) * Z_util[i, j] + len(A) * len(A[0]) * (1-Dhat_util[i, j]))

                

        # # constraint 4:
        # for i in range(len(nstar)):
        #     for j in range(len(mstar)):
        #         m_util.addConstr(Z_util[i,j] <= len(A) * D_util[i, j])
        #         m_util.addConstr(quicksum(X_util[k, l] for k in range(len(nstar)) for l in D[i]) - Z_util[i,j] <= len(A) * (1-D_util[i, j]))
        #         m_util.addConstr(Z_util[i, j]<= len(A) * quicksum(X_util[k, l] for k in range(len(nstar)) for l in D[i]))


        # # constraint 5: an agent who gets what he likes is not envious
        # for i in range(len(nstar)):
        #     for j in D[i]:
        #         m_util.addConstr(Z_util[i,j] == 0)
                
        # # #constraint 6: 
        # for i in range(len(nstar)):
        #     for j in range(len(mstar)):
        #         m_util.addConstr(X_util[i, j]>=0)
        #         m_util.addConstr(Z_util[i, j]>=0)
        #         m_util.addConstr(D_util[i, j]>=0)
        #         m_util.addConstr(D_util[i, j]<=1)
        #         m_util.addConstr(Dhat_util[i, j]>=0)
        #         m_util.addConstr(Dhat_util[i, j]<=1)

        # for i in range(len(nstar)):
        #     for j in range(len(mstar)):
        #         m_util.addConstr(X_util[i, j] <= len(A))
        #         m_util.addConstr(Z_util[i, j] <= len(A))

        


        # m_util.optimize()
        # # print(f"Optimal maxenvy value: {ml.objVal}")
        # UtilitarianEnvy.append(m_util.objVal)

        # enviousagentsinUHA= 0
        # maxenvyinUHA = 0
        # for i in range(len(nstar)):
        #     envyinUHA=0
        #     for j in range(len(mstar)): 
        #         if j not in D[i] and X_util[i,j].X > 0:  
        #             for k in D[i]:
        #                 for agents in range(len(nstar)):
        #                     if X_util[agents, k].X > 0:
        #                         enviousagentsinUHA += X_util[i, j].X
        #                         envyinUHA = envyinUHA + X_util[agents, k].X
        #                         if maxenvyinUHA < envyinUHA:
        #                             maxenvyinUHA = envyinUHA

        # EnviousAgentsinUHA.append(enviousagentsinUHA)
        # MaxEnvyinUHA.append(maxenvyinUHA)


        # UHAtime = time.time()-start2
    # OHAaverage.append(sum(EnviousAgents)/len(EnviousAgents))
    
    table.append([(Agents_Houses_Types[num][0], Agents_Houses_Types[num][1], Agents_Houses_Types[num][2]),sum(DummyHouses)/len(DummyHouses),
                sum(EnviousAgents)/len(EnviousAgents), sum(EnvyAmount)/len(EnvyAmount), sum(MaxEnvy)/len(MaxEnvy),
                sum(EgalitarianEnvy)/len(EgalitarianEnvy),sum(EnvyinEHA)/len(EnvyinEHA),sum(EnviousAgentsinEHA)/len(EnviousAgentsinEHA),
                # sum(UtilitarianEnvy)/len(UtilitarianEnvy), sum(MaxEnvyinUHA)/len(MaxEnvyinUHA), sum(EnviousAgentsinUHA)/len(EnviousAgentsinUHA),
                (sum(MaxAgentDegree)/len(MaxAgentDegree),sum(MaxHouseDegree)/len(MaxHouseDegree)), (round(OHAtime,2), round(EHAtime, 2))])

# print("-----------------")

print(tabulate(table, headers = ["(N, M, T)","#DummyHouse", "OHA","TotalEnvy(OHA)","MaxEnvy(OHA)","EHA","TotalEnvy(EHA)", "#Envious(EHA)", 
# "UHA", "MaxEnvyUHA", "#Envious(UHA)",
"(Agent,House)Degree", "Time(OHA/EHA)"]))

# print("UHAtime:", UHAtime)  

# X = [15, 30, 60, 90, 120]
# Y = np.linspace(0,5,70)

# plt.plot(X, OHAaverage[:5], color='r', label='agent types = 3')
# plt.plot(X, OHAaverage[5:10], color='g', label='agent types = 5')
# plt.plot(X, OHAaverage[10:15], color='b', label='agent types = 15')
# plt.xlabel("#Agents(=#Houses)")
# plt.ylabel("#Envious Agents")
# plt.legend()
# plt.show()


