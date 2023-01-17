#!/usr/bin/env python
# coding: utf-8

# In[8]:


#BFS
graph={'A':['B','C','D'],'B':['E'],'C':['D','E'],'D':[],'E':[]}
queue=[]
ans=[]
visited= set()
for s in list(graph.keys()):
    if s not in visited:
        queue.append(s)
        while(queue):
            s= queue.pop(0)
    if s in visited:
        continue
    visited.add(s)
    ans.append(s)
    for neighbours in graph[s]:
        if neighbours not in queue:
            queue.append(neighbours)
for a in ans:
    print(a)


# In[5]:


#DFS
graph ={ 'A':['B','C','D'], 'B':['E'],'C':['D'],'D':[],'E':['C']}
visited = set()
fnode= input("Enter first node")
def dfs(visited,graph,root):
    if root not in visited:
        print(root)
        visited.add(root)
        for i in graph[root]:
            dfs(visited,graph,i)

dfs(visited,graph,fnode)
        


# In[46]:


#Robot traversal
def path(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node
    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or graph[n] == None:
            pass
        else:
            for (m, weight) in nxt_states(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n == None:
            print('Path does not exist!')
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path to be taken found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('Path does not exist!')
    return None

def nxt_states(v):
    if v in graph:
        return graph[v]
    else:
        return None

def heuristic(n):
    H_dist = {
    'A': 11,
    'B': 6,
    'C': 99,
    'D': 1,
    'E': 7,
    'G': 0,
    }
    return H_dist[n]
graph = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1),('G', 9)],
    'C': None,
    'E': [('D', 6)],
    'D': [('G', 1)],
}
path('A', 'G')


# In[40]:


#TIC TAC TOE
def printboard(xState,zState):
    zero='X'if xState[0] else('O' if zState[0] else 0)
    one='X'if xState[1] else('O' if zState[1] else 1)
    two='X'if xState[2] else('O' if zState[2] else 2)
    three='X'if xState[3] else('O' if zState[3] else 3)
    four='X'if xState[4] else('O' if zState[4] else 4)
    five='X'if xState[5] else('O' if zState[5] else 5)
    six='X'if xState[6] else('O' if zState[6] else 6)
    seven='X'if xState[7] else('O' if zState[7] else 7)
    eight='X'if xState[8] else('O' if zState[8] else 8)
    print(f"{zero}|{one}|{two}")
    print(f"--|--|--")
    print(f"{three}|{four}|{five}")
    print(f"--|--|--")
    print(f"{six}|{seven}|{eight}")
    
    
def checkwin(xState,zState):
    wins=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for win in wins:
        if (xState[win[0]]+xState[win[1]]+xState[win[2]])==3:
            print("X WON")
            return 1
        if (zState[win[0]]+zState[win[1]]+zState[win[2]])==3:
            print("Zwon")
            return 0
        return -1

xState=[0,0,0,0,0,0,0,0,0]
zState=[0,0,0,0,0,0,0,0,0]
turn =1
while(True):
    printboard(xState,zState)
    if (turn ==1):
        print("x chance")
        value=int(input())
        xState[value]=1
    else:
        print("O 's chance")
        value = int(input())
        zState[value]=1
    cwin=checkwin(xState,zState)
    if cwin!=-1:
        print("Matchover")
        break
    turn = 1-turn
    



# In[37]:


#TSP
from sys import maxsize
v = 4

def travelling_salesman_function(graph, s):
    vertex = []
    for i in range(v):
        if i != s:
            vertex.append(i)

    min_path = maxsize
    while True:
        current_cost = 0
        k = s
        for i in range(len(vertex)):
            current_cost += graph[k][vertex[i]]
            k = vertex[i]
        current_cost += graph[k][s]
        min_path = min(min_path, current_cost)

        if not next_perm(vertex):
            break
    return min_path

def next_perm(l):
    n = len(l)
    i = n-2

    while i >= 0 and l[i] > l[i+1]:
        i -= 1
    
    if i == -1:
        return False

    j = i+1
    while j < n and l[j] > l[i]:
        j += 1

    j -= 1

    l[i], l[j] = l[j], l[i]
    left = i+1
    right = n-1

    while left < right:
        l[left], l[right] = l[right], l[left]
        left += 1
        right -= 1
    return True

graph = [[0,10,15,20], [10,0,35,25], [15,35,0,30], [20,25,30,0]]
s = 0
res = travelling_salesman_function(graph,s)
print(res)


# In[24]:


# Taking number of queens as input from user
print ("Enter the number of queens")
N = int(input())
# here we create a chessboard
# NxN matrix with all elements set to 0
board = [[0]*N for _ in range(N)]
def attack(i, j):
    #checking vertically and horizontally
    for k in range(0,N):
        if board[i][k]==1 or board[k][j]==1:
            return True
    #checking diagonally
    for k in range(0,N):
        for l in range(0,N):
            if (k+l==i+j) or (k-l==i-j):
                if board[k][l]==1:
                    return True
    return False
def N_queens(n):
    if n==0:
        return True
    for i in range(0,N):
        for j in range(0,N):
            if (not(attack(i,j))) and (board[i][j]!=1):
                board[i][j] = 1
                if N_queens(n-1)==True:
                    return True
                board[i][j] = 0
    return False
N_queens(N)
for i in board:
    print (i)


# In[27]:


def TOI(n , s, d, a): 
    if n==1:
        print ("Move disk 1 from src",s,"to dst",d)
        return
    TOI(n-1, s, a, d)
    print ("Move disk",n,"from src",s,"to dst",d) 
    TOI(n-1, a, d, s)


n = 4 
TOI(3,'A','B','C')


# In[28]:


#HILL CLIMBING
import random
from random import shuffle
import numpy as np
class State:
    def __init__(self, state):
        self.state = state
    def evaluation(self):
        h = 0
        temp = 0
        for i in self.state:
            temp = self.state.count(i)
            if temp > 1:
                h += 1
        for i in range(0, len(self.state)):
            for j in range(0, len(self.state)):
                if j > i:
                    if abs(i - j) == abs(self.state[i] - self.state[j]):
                        h += 1
        return h
    def neighbor(self):
        neighbors = {}
        for i in range(0, len(self.state)):
            for j in range(0, len(self.state)):
                if j != self.state[i]:
                    temp = self.state.copy()
                    temp[i] = j
                    temp = State(temp)
                    neighbors[(i, j)] = temp.evaluation()
        best_neighbors = []
        best_h = self.evaluation()
        for i, h in neighbors.items():
            if h < best_h:
                best_h = h
            if h == best_h:
                best_neighbors.append(i)
        if len(best_neighbors) > 0:
            random_index = random.randint(0, len(best_neighbors) - 1)
            self.state[best_neighbors[random_index][0]] = best_neighbors[random_index][1]
        return State(self.state)
initial_state = [0, 0, 0, 0, 0, 0, 0, 0]
def hill_climbing(initial_state):
    current_state = State(initial_state)
    while True:
        best_neighbor = current_state.neighbor()
        if best_neighbor.evaluation() >= current_state.evaluation():
            return current_state.state
        current_state = best_neighbor
    return current_state
def random_restart(initial_state):
    state = State(initial_state)
    count = 0
    while State(initial_state).evaluation() > 0 and count < 10000:
        shuffle(initial_state)
        state = hill_climbing(initial_state)
        count += 1
    print("Number of states explored: ",count)
    return state
solution = random_restart(initial_state)
def board(solution):
    matrix = np.zeros([8,8], dtype=int)
    matrix = matrix.tolist()
    for item in solution:
        for i in range(len(solution)):
            if i == item:
                for j in range(len(solution)):
                    if j == solution.index(item):
                        matrix[i][j] = 1
    l =[]
    for i in range(1, len(solution)+1):
        l.append(i)
    print(matrix)
board(solution)


# In[38]:


#8PUZZLE
import copy
 
# Importing the heap functions from python
# library for Priority Queue
from heapq import heappush, heappop
 
# This variable can be changed to change
# the program from 8 puzzle(n=3) to 15
# puzzle(n=4) to 24 puzzle(n=5)...
n = 3
 
# bottom, left, top, right
row = [ 1, 0, -1, 0 ]
col = [ 0, -1, 0, 1 ]
 
# A class for Priority Queue
class priorityQueue:
     
    # Constructor to initialize a
    # Priority Queue
    def __init__(self):
        self.heap = []
 
    # Inserts a new key 'k'
    def push(self, k):
        heappush(self.heap, k)
 
    # Method to remove minimum element
    # from Priority Queue
    def pop(self):
        return heappop(self.heap)
 
    # Method to know if the Queue is empty
    def empty(self):
        if not self.heap:
            return True
        else:
            return False
 
# Node structure
class node:
     
    def __init__(self, parent, mat, empty_tile_pos,
                 cost, level):
                      
        # Stores the parent node of the
        # current node helps in tracing
        # path when the answer is found
        self.parent = parent
 
        # Stores the matrix
        self.mat = mat
 
        # Stores the position at which the
        # empty space tile exists in the matrix
        self.empty_tile_pos = empty_tile_pos
 
        # Stores the number of misplaced tiles
        self.cost = cost
 
        # Stores the number of moves so far
        self.level = level
 
    # This method is defined so that the
    # priority queue is formed based on
    # the cost variable of the objects
    def __lt__(self, nxt):
        return self.cost < nxt.cost
 
# Function to calculate the number of
# misplaced tiles ie. number of non-blank
# tiles not in their goal position
def calculateCost(mat, final) -> int:
     
    count = 0
    for i in range(n):
        for j in range(n):
            if ((mat[i][j]) and
                (mat[i][j] != final[i][j])):
                count += 1
                 
    return count
 
def newNode(mat, empty_tile_pos, new_empty_tile_pos,
            level, parent, final) -> node:
                 
    # Copy data from parent matrix to current matrix
    new_mat = copy.deepcopy(mat)
 
    # Move tile by 1 position
    x1 = empty_tile_pos[0]
    y1 = empty_tile_pos[1]
    x2 = new_empty_tile_pos[0]
    y2 = new_empty_tile_pos[1]
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]
 
    # Set number of misplaced tiles
    cost = calculateCost(new_mat, final)
 
    new_node = node(parent, new_mat, new_empty_tile_pos,
                    cost, level)
    return new_node
 
# Function to print the N x N matrix
def printMatrix(mat):
     
    for i in range(n):
        for j in range(n):
            print("%d " % (mat[i][j]), end = " ")
             
        print()
 
# Function to check if (x, y) is a valid
# matrix coordinate
def isSafe(x, y):
     
    return x >= 0 and x < n and y >= 0 and y < n
 
# Print path from root node to destination node
def printPath(root):
     
    if root == None:
        return
     
    printPath(root.parent)
    printMatrix(root.mat)
    print()
 
# Function to solve N*N - 1 puzzle algorithm
# using Branch and Bound. empty_tile_pos is
# the blank tile position in the initial state.
def solve(initial, empty_tile_pos, final):
     
    # Create a priority queue to store live
    # nodes of search tree
    pq = priorityQueue()
 
    # Create the root node
    cost = calculateCost(initial, final)
    root = node(None, initial,
                empty_tile_pos, cost, 0)
 
    # Add root to list of live nodes
    pq.push(root)
 
    # Finds a live node with least cost,
    # add its children to list of live
    # nodes and finally deletes it from
    # the list.
    while not pq.empty():
 
        # Find a live node with least estimated
        # cost and delete it form the list of
        # live nodes
        minimum = pq.pop()
 
        # If minimum is the answer node
        if minimum.cost == 0:
             
            # Print the path from root to
            # destination;
            printPath(minimum)
            return
 
        # Generate all possible children
        for i in range(4):
            new_tile_pos = [
                minimum.empty_tile_pos[0] + row[i],
                minimum.empty_tile_pos[1] + col[i], ]
                 
            if isSafe(new_tile_pos[0], new_tile_pos[1]):
                 
                # Create a child node
                child = newNode(minimum.mat,
                                minimum.empty_tile_pos,
                                new_tile_pos,
                                minimum.level + 1,
                                minimum, final,)
 
                # Add child to list of live nodes
                pq.push(child)
 
# Driver Code
 
# Initial configuration
# Value 0 is used for empty space
initial = [ [ 1, 2, 3 ],
            [ 5, 6, 0 ],
            [ 7, 8, 4 ] ]
 
# Solvable Final configuration
# Value 0 is used for empty space
final = [ [ 1, 2, 3 ],
          [ 5, 8, 6 ],
          [ 0, 7, 4 ] ]
 
# Blank tile coordinates in
# initial configuration
empty_tile_pos = [ 1, 2 ]
 
# Function call to solve the puzzle
solve(initial, empty_tile_pos, final)


# In[37]:


#JUG
def action(jug1, jug2):
    capacity1, capacity2, fill = 3, 5, 4 #Change maximum capacity and final capacity 
    print("%d\t%d" % (jug1, jug2))
    if jug2 == fill: 
        return
    elif jug2 == capacity2: 
        action(0, jug1)
    elif jug1 != 0 and jug2 == 0: 
        action(0, jug1)
    elif jug1 == fill: 
        action(jug1, 0)
    elif jug1 < capacity1: 
        action(capacity1, jug2)
    elif jug1 < (capacity2-jug2): 
        action(0, (jug1+jug2))
    else:
        action(jug1-(capacity2-jug2), (capacity2-jug2)+jug2)

print("JUG1\tJUG2") 
action(0, 0)


# In[ ]:




