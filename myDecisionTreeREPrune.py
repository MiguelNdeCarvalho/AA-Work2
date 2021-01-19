from sklearn.model_selection import train_test_split
import argparse
import math
import numpy
from Node import Node
import time

class myDecisionTreeREPrune: 
    
    #construtores
    def __init__(self, criterion, prune):
        
        self.criterion=criterion
        self.prune=prune
        print(f"Chosen criterion: {self.criterion} with prune: {self.prune}")
        self.root = Node()

    #metodos
    def fit(self,x,y,attributeList):
        
        is_homegenus,homo_classe = homogeneous(y)
        if is_homegenus: 
            self.root = makeLeaf(homo_classe)
        else:
            globalTotalValues = globalTotalCount(y)

            if self.criterion == "entropy":
                globalTotal = entropyCalc(globalTotalValues)
            elif self.criterion == "gini":
                globalTotal = giniCalc(globalTotalValues)
            elif self.criterion == "error":
                globalTotal = errorCalc(globalTotalValues)

            rootNode = chooseNode(self.criterion, attributeList,x,y,globalTotalValues, globalTotal) 
            self.root = Node()
            self.root.set_Data(rootNode)

            valuesLeafs,classeLeafs = checkLeaf(rootNode, attributeList, x, y)
            not_leafs,sons_order = valuesNotToLeafs(valuesLeafs,x,self.root.get_Data(),attributeList)

            setSons(self.root,sons_order,not_leafs,valuesLeafs,classeLeafs)
            self.root.set_Sons(sons_order)
            
            for i in range(len(self.root.sons)):
                if self.root.sons[i].get_Data()=="":
                    attributeList,pos = removeAttribute(attributeList,rootNode)
                    x,y,save_rows,save_value = update_data(x,y,self.root.sons[i],sons_order[i]) 
                    if len(attributeList[0]) == 0:
                        new_son = Node()
                        new_son.set_Data(most_of(y))
                        self.root.sons[i] = new_son
                    else:
                        grow_tree(self,self.root.sons[i],x,y,attributeList)
                    x,y = applySaves(x,y,save_rows,save_value,pos)
                    attributeList = appendAttribute(attributeList,rootNode,pos)

        return 0




    def score(self,x,y,attributeList):
        
        total = len(x)
        count_hit = 0

        for row in range(len(x)): 

            result = iterate_for(self,x[row,0:],y[row],attributeList)
            if result == "hit":
                count_hit+=1

        return count_hit/total *100


        
def iterate_for(self,x_object,y_object,attributeList):
    
    node_atual = self.root

    while not node_atual.is_leaf():
        
        node_data = node_atual.get_Data()
        node_sons = node_atual.get_Sons()

        for i in range(len(attributeList[0])):
            if attributeList[0][i] == node_data:
                break

        for x in range(len(node_atual.get_Sons())):
            if x_object[i] == node_sons[x]:
                break

        node_atual = node_atual.sons[x]

    if node_atual.get_Data() == y_object:
        return "hit"
    
    return "miss" 

def most_of(y):
    
    possible = []
    aux = []
    for data in y:
        if data not in possible:
            possible.append(data)
            aux.append(0)
    for data in y:
        count = 0
        for value in possible:
            if data == value:
                aux[count] +=1
            count +=1

    max_y_index = 0
    for i in range(len(aux)):
        if aux[i] > aux[max_y_index]:
            max_y_index = i
    
   #print(aux,possible,possible[max_y_index])
    return possible[max_y_index]

def grow_tree(self,node,x,y,attributeList):

    globalTotalValues = globalTotalCount(y)
    globalTotal=entropyCalc(globalTotalValues)

    new_branch = chooseNode(self.criterion, attributeList,x,y,globalTotalValues, globalTotal) #argument

    valuesLeafs,classeLeafs = checkLeaf(new_branch, attributeList, x,y)

    node.root = Node()
    node.set_Data(new_branch)

    not_leafs,sons_order = valuesNotToLeafs(valuesLeafs,x,node.get_Data(),attributeList)
    node.set_Sons(sons_order)

    setSons(node,sons_order,not_leafs,valuesLeafs,classeLeafs)
    
    for i in range(len(node.sons)):

        if node.sons[i].get_Data()=="":
            attributeList,pos = removeAttribute(attributeList,new_branch)
            x,y, save_rows,save_value = update_data(x,y,node.sons[i],sons_order[i])
            if len(attributeList[0]) == 0:
                new_son = Node()
                new_son.set_Data(most_of(y))
                node.sons[i] = new_son
            else:
                grow_tree(self,node.sons[i],x,y,attributeList)
            x,y = applySaves(x,y,save_rows,save_value,pos)
            attributeList = appendAttribute(attributeList,new_branch,pos)

def update_data(x,y,attri,value):

    new_x = []
    new_y = []
    row = []

    save_value = value
    save_rows = []

    if len(value) == 0:
        return

    count=0
    for row in zip(x,y):
        
        found = False
        for i in row[0]:
            if i == value:
                found = True
            
        if found:
            new_x.append(list(row[0])) 
            new_y.append(row[1])
        else:
            new_row = []
            new_row.append(count)
            new_row.append(list(row[0]))
            new_row.append(row[1])
            save_rows.append(list(new_row))
        count+=1

    for row in new_x:
        row.remove(value)

    x = list(new_x)
    y = list(new_y)

    return x,y,save_rows,save_value

def applySaves(x,y,save_rows,save_value,value_pos):
    
    save = []

    for row in x:
        new_x_row = []
        for values in range(len(row)):
            if values == value_pos:
                new_x_row.append(save_value)
                new_x_row.append(row[values])
            else:
                new_x_row.append(row[values])
        save.append(list(new_x_row))


    x = save
    new_x = []
    new_y = []

    count = 0
    end = len(save_rows) + len(x) -1

    for n in range(end):

        if len(save_rows) != 0 and save_rows[0][0] == count:
            new_x.append(list(save_rows[0][1]))
            new_y.append(save_rows[0][2])
            save_rows.pop(0)
        else:
            new_x.append(list(x.pop(0)))
            new_y.append(y.pop(0))
        count+=1

    x=new_x
    y=new_y

    return x,y


def appendAttribute(attributeList,attri,pos):
    
    aux = []
    for row in attributeList:
        for i in range(len(row)):
            if i == pos:
                aux.append(attri)
                aux.append(row[i])
            else: 
                aux.append(row[i])

    return numpy.array([aux])

def removeAttribute(attributeList,attri):

    aux = []
    save = None

    for row in attributeList:
        count = 0
        for pos in row:
            if pos not in aux and pos != attri:
                aux.append(pos)
            else:
                save = count
        count+=1

    return numpy.array([aux]),save

def PrintTree(node):
    
    if node == None:
        pass
    
    print(node.get_Data())

    if not node.is_leaf():
        for sons in range(len(node.sons)):
            PrintTree(node.sons[sons])
        


def setSons(node,sons_order,valuesNotLeafs,valuesLeafs,classeLeafs):
    
    for sons in range(len(sons_order)):
        
        if sons_order[sons] in valuesLeafs:
            for pos in range(len(valuesLeafs)):
                if sons_order[sons] == valuesLeafs[pos]:
                    node.sons.append(makeLeaf(classeLeafs[pos]))
        
        if sons_order[sons] in valuesNotLeafs:
            for pos in range(len(valuesNotLeafs)):
                if sons_order[sons] == valuesNotLeafs[pos]:
                    no = Node()
                    node.sons.append(no)


def makeLeaf(classe):
    
    no = Node()
    no.set_Data(classe)

    return no


def valuesNotToLeafs(valuesLeafs,x,atr,attributeList):
    atribute_values = getValues(x,attributeList,atr)

    aux = [] 

    for atri_val in atribute_values:
        if atri_val not in valuesLeafs:
            aux.append(atri_val)

    return aux,atribute_values

def isLeaf(attribute, attributeList, value, xdata, ydata): #checks if that value only has the same value
    '''
    Função que verifica se um atributo é uma folha, ou seja, só já tem uma classe
    1º - verifica quais as classes existentes para esse atributo
    2º - caso seja só 1, diz que é um Leaf.
    '''
    attributePos = numpy.where(attributeList == attribute)
    aux = []
    for x,y in zip(xdata, ydata):
        if x[attributePos[1][0]] == value and y not in aux:
            aux.append(y)
    if len(aux) == 1:
        return True,aux[0]
    return False,[]

def checkLeaf(rootNode, attributeList, xdata, ydata):
    '''
    Verifica se o atributo principal (rootNode) apresenta leafs.
    Retorna um array com os values sem os leafs
    '''
    aux=[]
    aux_class = []
    values = getValues(xdata, attributeList, rootNode)
    for value in values:
        condition,classe = isLeaf(rootNode, attributeList, value, xdata, ydata)
        if condition:
            aux.append(value)
            aux_class.append(classe)
    return aux,aux_class

def globalTotalCount(ydata):
    '''
    Esta função retorna num array quantas vezes aparece cada class
    1º For - adiciona no array possible todas as classes possiveis
    2º For - percorre e verifica quantas vezes aparece cada posição possivel
    '''
    possible = []
    aux = []
    for data in ydata:
        if data not in possible:
            possible.append(data)
            aux.append(0)
    for data in ydata:
        count = 0
        for value in possible:
            if data == value:
                aux[count] +=1
            count +=1
    return aux

def entropyCalc(array):
    '''
    recebe como argumento o numero de vezes que cada classe aparece e calcula a entropia
    for - Calcula a Entropia em si
    '''

    total, result = sum(array),0
    for pos in array:
        result -= (pos/total * math.log2(pos/total))
    return result

def giniCalc(array):
    '''
    recebe como argumento o numero de vezes que cada classe aparece e calcula o gini 
    for - Calcula o Gini em si
    '''
    total, result = sum(array),1
    for pos in array:
        result -= (pos/total) ** 2
    return result

def errorCalc(array):
    '''
    recebe como argumento o numero de vezes que cada classe aparece e calcula o erro
    for - Calcula o erro em si
    '''
    total, aux = sum(array), []
    for pos in array:
        aux.append(pos/total)
    result = 1 - max(aux) 
    return result


def getValues(data, attributeList, attribute):

    '''
    Função que retorna um array com os valores de um dado atributo
    '''
    values = []
    attributePos = numpy.where(attributeList == attribute)

    for value in data:
        #print(f"Value: {value} Pos: {attributePos[1][0]}")
        if value[attributePos[1][0]] not in values:
            values.append(value[attributePos[1][0]])
    return values

def valueCount(attribute, attributeList, value, xdata, ydata):
    '''
    Função que conta o número de classes correspondentes a um value de um atributo
    1º For - adiciona no array possible todas as classes possiveis
    2º For - percorre e verifica quantas vezes aparece cada posição possivel
    '''
    possible = []
    aux = []
    attributePos = numpy.where(attributeList == attribute)
    for x,y in zip(xdata,ydata):
        if x[attributePos[1][0]] == value and y not in possible:
            possible.append(y)
            aux.append(0)
    for x,y in zip(xdata,ydata):
        count = 0
        for i in possible:
            if x[attributePos[1][0]] == value and y == i:
                aux[count]+=1
            count +=1
    return aux

def calculateGain(criterion,attribute, attributeList, xdata, ydata, globalTotalValues ,globalTotal):
    '''
    Função que calcula:
    1º - entropia para cada value de um atributo
    2º - a informação entropia de um atributo
    3º - o ganho (gain) de um atributo
    retorna o ganho de uma class
    '''
    values = getValues(xdata,attributeList, attribute)
    total = 0

    if criterion == "entropy":
        for value in values:
            values = valueCount(attribute, attributeList, value, xdata, ydata)
            entropy = entropyCalc(values)
            total += (sum(values) / sum(globalTotalValues)) * entropy    
        gain = globalTotal - total
        return gain
    elif criterion == "gini":
        for value in values:
            values = valueCount(attribute, attributeList, value, xdata, ydata)
            gini = giniCalc(values)
            total += (sum(values) / sum(globalTotalValues)) * gini
        gain = globalTotal - total
        return gain
    elif criterion == "error":
        for value in values:
            values = valueCount(attribute, attributeList, value, xdata, ydata)
            error = errorCalc(values)
            total += (sum(values) / sum(globalTotalValues)) * error
        gain = globalTotal - total
        return gain

def chooseNode(criterion,attributeList,xdata, ydata,globalTotalValues, globalTotal):
    '''
    Função que recebe a lista de attributes e calcula o ganho de cada e retorna o attributo com maior ganho,
    o atributo principal
    '''
    aux = []
    for attribute in attributeList[0]:
        aux.append(calculateGain(criterion,attribute, attributeList, xdata, ydata, globalTotalValues, globalTotal))
        index = aux.index(max(aux))
    return attributeList[0][index]


def entropyRootValues(rootNode, attributeList, valuesWithoutLeafs, xdata, ydata):
    '''
    Função que devolve a entropia de cada value do Atributo Principal (RootNode)
    '''
    aux=[]
    for attribute in valuesWithoutLeafs:
        count = valueCount(rootNode, attributeList, attribute, xdata, ydata)
        aux.append(entropyCalc(count))
    return aux

def homogeneous(y):     
    

    ref_value = y[0][0]


    for samples in range(len(y)):
        if ref_value != y[samples][0]:
            return False,""

    return True,ref_value  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()        
    parser.add_argument('-f', "-file", action='store', dest='file_location', help='Path to the file that contains the data', required=True)
    parser.add_argument('-c', '-criterion', action='store', dest='criterion', help='Choose criterion, it can be entropy, gini or error. By default: Entropy', default='entropy')
    parser.add_argument('-p', '-prune', action='store_true', dest='prune', help='Set prune. By default: False', default=False)
    results = parser.parse_args()


    data=numpy.genfromtxt("dados/led-train.csv", delimiter=",", dtype=None, encoding=None)
    x_train=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
    y_train=data[1:,-1]      # classe: da segunda à ultima linha, só última coluna
    attributeList=data[:1,:-1]

    data=numpy.genfromtxt("dados/led-test.csv", delimiter=",", dtype=None, encoding=None)
    x_test=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
    y_test=data[1:,-1]   
    
    tree = myDecisionTreeREPrune(criterion=results.criterion, prune=results.prune)
    print("Fitting.....")
    tree.fit(x_train,y_train,attributeList)
    print("Fit Done")
    print("Scoring.....")
    print("Score: ",tree.score(x_test,y_test,attributeList))

