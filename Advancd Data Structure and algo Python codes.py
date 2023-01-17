#!/usr/bin/env python
# coding: utf-8

# In[5]:


#BubbleSORT
def bubble(arr):
    n = len(arr)
    for i in range (n):
        for j in range(0, n-i-1):
            if arr[j]>arr[j+1]:
                arr[j], arr[j+1]= arr[j+1], arr[j]
arr= [2,4,1,5,13,8]
bubble(arr)
n= len(arr)
for i in range(n):
    print(arr[i], end = " ")


# In[7]:


#Insertion Sort
def insertionsort(arr):
    n = len(arr)
    for i in range (1, len(arr)):
        j = i-1
        temp = arr[i]
        while j>=0 and temp<arr[j]:
            arr[j+1]= arr[j]
            j=j-1
        arr[j+1] = temp
arr= [9,8,5,6,7,1]
insertionsort(arr)
print(arr)
            


# In[9]:


#selection sort
def selectionsort(arr):
    n = len(arr)
    for i in range(n):
        min_idx= i
        for j in range(i+1, n):
                if arr[min_idx]> arr[j]:
                    min_idx= j
        arr[i],arr[min_idx] = arr[min_idx], arr[i]
    return arr
arr = [5,7,2,3,22,11,10]
selectionsort(arr)
print(arr)


# In[10]:


#Quick sort
def partition (a, start, end):  
    i = (start - 1)  
    pivot = a[end] # pivot element  
      
    for j in range(start, end):  
        # If current element is smaller than or equal to the pivot  
        if (a[j] <= pivot):  
            i = i + 1  
            a[i], a[j] = a[j], a[i]  
      
    a[i+1], a[end] = a[end], a[i+1]  
  
    return (i + 1)  
# function to implement quick sort   
def quick(a, start, end): # a[] = array to be sorted, start = Starting index, end = Ending index      
    if (start < end):  
        p = partition(a, start, end) # p is partitioning index  
        quick(a, start, p - 1)  
        quick(a, p + 1, end)  
  
          
def printArr(a): # function to print the array  
    for i in range(len(a)):  
        print (a[i], end = " ")  
      
a = [68, 13, 1, 49, 58]  
print("Before sorting array elements are - ")  
printArr(a)  
quick(a, 0, len(a)-1)  
print("\nAfter sorting array elements are - ")  
printArr(a) 
        


# In[11]:


#Merge sort
def mergeSort(arr):
    if len(arr) > 1:
 
         # Finding the mid of the array
        mid = len(arr)//2
 
        # Dividing the array elements
        L = arr[:mid]
 
        # into 2 halves
        R = arr[mid:]
 
        # Sorting the first half
        mergeSort(L)
 
        # Sorting the second half
        mergeSort(R)
 
        i = j = k = 0
 
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
 
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
 
def printList(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print() 
# Driver Code
if __name__ == '__main__':
    arr = [12, 11, 13, 5, 6, 7]
    print("Given array is", end="\n")
    printList(arr)
    mergeSort(arr)
    print("Sorted array is: ", end="\n")
    printList(arr)


# In[17]:


class Node:
    def __init__ (self, data):
        self.data = data
        self.next = None
a = Node(13)
b= Node(15)
a.next=b
print(a.data)
print(b.data)
print(a.next.data)


# In[18]:


# Linked list operations in Python


# Create a node
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None

    # Insert at the beginning
    def insertAtBeginning(self, new_data):
        new_node = Node(new_data)

        new_node.next = self.head
        self.head = new_node

    # Insert after a node
    def insertAfter(self, prev_node, new_data):

        if prev_node is None:
            print("The given previous node must inLinkedList.")
            return

        new_node = Node(new_data)
        new_node.next = prev_node.next
        prev_node.next = new_node

    # Insert at the end
    def insertAtEnd(self, new_data):
        new_node = Node(new_data)

        if self.head is None:
            self.head = new_node
            return

        last = self.head
        while (last.next):
            last = last.next

        last.next = new_node

    # Deleting a node
    def deleteNode(self, position):

        if self.head is None:
            return

        temp = self.head

        if position == 0:
            self.head = temp.next
            temp = None
            return

        # Find the key to be deleted
        for i in range(position - 1):
            temp = temp.next
            if temp is None:
                break

        # If the key is not present
        if temp is None:
            return

        if temp.next is None:
            return

        next = temp.next.next

        temp.next = None

        temp.next = next

    # Search an element
    def search(self, key):

        current = self.head

        while current is not None:
            if current.data == key:
                return True

            current = current.next

        return False

    # Sort the linked list
    def sortLinkedList(self, head):
        current = head
        index = Node(None)

        if head is None:
            return
        else:
            while current is not None:
                # index points to the node next to current
                index = current.next

                while index is not None:
                    if current.data > index.data:
                        current.data, index.data = index.data, current.data

                    index = index.next
                current = current.next

    # Print the linked list
    def printList(self):
        temp = self.head
        while (temp):
            print(str(temp.data) + " ", end="")
            temp = temp.next


if __name__ == '__main__':

    llist = LinkedList()
    llist.insertAtEnd(1)
    llist.insertAtBeginning(2)
    llist.insertAtBeginning(3)
    llist.insertAtEnd(4)
    llist.insertAfter(llist.head.next, 5)

    print('linked list:')
    llist.printList()

    print("\nAfter deleting an element:")
    llist.deleteNode(3)
    llist.printList()

    print()
    item_to_find = 3
    if llist.search(item_to_find):
        print(str(item_to_find) + " is found")
    else:
        print(str(item_to_find) + " is not found")

    llist.sortLinkedList(llist.head)
    print("Sorted List: ")
    llist.printList()


# In[20]:


# Circular Queue implementation in Python
class MyCircularQueue():

    def __init__(self, k):
        self.k = k
        self.queue = [None] * k
        self.head = self.tail = -1

    # Insert an element into the circular queue
    def enqueue(self, data):

        if ((self.tail + 1) % self.k == self.head):
            print("The circular queue is full\n")

        elif (self.head == -1):
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            self.tail = (self.tail + 1) % self.k
            self.queue[self.tail] = data

    # Delete an element from the circular queue
    def dequeue(self):
        if (self.head == -1):
            print("The circular queue is empty\n")

        elif (self.head == self.tail):
            temp = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return temp
        else:
            temp = self.queue[self.head]
            self.head = (self.head + 1) % self.k
            return temp

    def printCQueue(self):
        if(self.head == -1):
            print("No element in the circular queue")

        elif (self.tail >= self.head):
            for i in range(self.head, self.tail + 1):
                print(self.queue[i], end=" ")
            print()
        else:
            for i in range(self.head, self.k):
                print(self.queue[i], end=" ")
            for i in range(0, self.tail + 1):
                print(self.queue[i], end=" ")
            print()


# Your MyCircularQueue object will be instantiated and called as such:
obj = MyCircularQueue(5)
obj.enqueue(1)
obj.enqueue(2)
obj.enqueue(3)
obj.enqueue(4)
obj.enqueue(5)
print("Initial queue")
obj.printCQueue()

obj.dequeue()
print("After removing an element from the queue")
obj.printCQueue()


# In[25]:


#BST
# Python program to demonstrate
# insert operation in binary search tree
 
# A utility class that represents
# an individual node in a BST
 
 
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
 
# A utility function to insert
# a new node with the given key
 
 
def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if root.val == key:
            return root
        elif root.val < key:
            root.right = insert(root.right, key)
        else:
            root.left = insert(root.left, key)
    return root
 
# A utility function to do inorder tree traversal
def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)
 
def preorder(root):
     if root:
        
        print(root.val)
        preorder(root.left)
        preorder(root.right)
    

def postorder(root):
    if root:
        
        postorder(root.left)
        preorder(root.right)
        print(root.val)
    
 
r = Node(50)
r = insert(r, 30)
r = insert(r, 20)
r = insert(r, 40)
r = insert(r, 70)
r = insert(r, 60)
r = insert(r, 80)
 
# Print inoder traversal of the BST
inorder(r)
preorder(r)
postorder(r)


# In[28]:


#BFS
graph = {
'A' : ['B','C'],
'B' : ['D', 'E'],
'C' : ['F','G'], 'D' : [],
'E' : [],
'F' : [],
'G' : [],
}
stack = [] 
visited = set() 
ans = []
for s in list(graph.keys()): 
    if s not in visited:
        stack.append(s) 
        while(stack):
            s = stack.pop(0)
            if s in visited:
                continue 
            visited.add(s) 
            ans.append(s)
            for neighbours in graph[s]:
                if neighbours not in stack:
                    stack.append(neighbours)


for a in ans: 
    print(a)


# In[29]:


#DFS
graph = {
'A' : ['B','C'],
'B' : ['D', 'E'],
'C' : ['F'],
'D' : [],
'E' : ['F'],
'F' : [],
'G' : ['H'], 'H' : []
}
stack = [] 
visited = set() 
ans = []
for s in list(graph.keys()): 
    if s not in visited:
        stack.append(s) 
        while(stack):
            s = stack.pop() 
            if s in visited:
                continue 
            visited.add(s) 
            ans.append(s)
            for neighbours in graph[s]:
                if neighbours not in stack:
                    stack.append(neighbours)


for a in ans: 
    print(a)


# In[30]:


#Binary search
def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
 
    while low <= high:
 
        mid = (high + low) // 2
 
        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1
 
        # means x is present at mid
        else:
            return mid
 
    # If we reach here, then the element was not present
    return -1
 
 
# Test array
arr = [ 2, 3, 4, 10, 40 ]
x = 10
 
# Function call
result = binary_search(arr, x)
 
if result != -1:
    print("Element is present at index", str(result))
else:
    print("Element is not present in array")


# In[31]:


class Stack:
    def __init__(self):
        self.__data=[]
    def push(self,item):
        self.__data.append(item)
    def pop(self):
        if self.isEmpty():
            print("Hey!Stack is Empty!!")
            return
        return self.__data.pop()
    
    def top(self):
        if self.isEmpty():
            print("Hey!Stack is Empty!!")
            return
        return self.__data[len(self.__data)-1]
        
    def size(self):
        return len(self.__data)
        
    def isEmpty(self):
        return self.size()==0

s=Stack()
s.push(12)
s.push(13)
s.push(15)
while s.isEmpty() is False:
    print(s.pop())
s.top()


# In[32]:


#AVL TREE
class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1
 
# AVL tree class which supports the
# Insert operation
class AVL_Tree(object):
 
    # Recursive function to insert key in
    # subtree rooted with node and returns
    # new root of subtree.
    def insert(self, root, key):
     
        # Step 1 - Perform normal BST
        if not root:
            return TreeNode(key)
        elif key < root.val:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)
 
        # Step 2 - Update the height of the
        # ancestor node
        root.height = 1 + max(self.getHeight(root.left),
                           self.getHeight(root.right))
 
        # Step 3 - Get the balance factor
        balance = self.getBalance(root)
 
        # Step 4 - If the node is unbalanced,
        # then try out the 4 cases
        # Case 1 - Left Left
        if balance > 1 and key < root.left.val:
            return self.rightRotate(root)
 
        # Case 2 - Right Right
        if balance < -1 and key > root.right.val:
            return self.leftRotate(root)
 
        # Case 3 - Left Right
        if balance > 1 and key > root.left.val:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root)
 
        # Case 4 - Right Left
        if balance < -1 and key < root.right.val:
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root)
 
        return root
 
    def leftRotate(self, z):
 
        y = z.right
        T2 = y.left
 
        # Perform rotation
        y.left = z
        z.right = T2
 
        # Update heights
        z.height = 1 + max(self.getHeight(z.left),
                         self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                         self.getHeight(y.right))
 
        # Return the new root
        return y
 
    def rightRotate(self, z):
 
        y = z.left
        T3 = y.right
 
        # Perform rotation
        y.right = z
        z.left = T3
 
        # Update heights
        z.height = 1 + max(self.getHeight(z.left),
                        self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                        self.getHeight(y.right))
 
        # Return the new root
        return y
 
    def getHeight(self, root):
        if not root:
            return 0
 
        return root.height
 
    def getBalance(self, root):
        if not root:
            return 0
 
        return self.getHeight(root.left) - self.getHeight(root.right)
 
    def preOrder(self, root):
 
        if not root:
            return
 
        print("{0} ".format(root.val), end="")
        self.preOrder(root.left)
        self.preOrder(root.right)
 
 
# Driver program to test above function
myTree = AVL_Tree()
root = None
 
root = myTree.insert(root, 10)
root = myTree.insert(root, 20)
root = myTree.insert(root, 30)
root = myTree.insert(root, 40)
root = myTree.insert(root, 50)
root = myTree.insert(root, 25)
 
"""The constructed AVL Tree would be
            30
           /  \
         20   40
        /  \     \
       10  25    50"""
 
# Preorder Traversal
print("Preorder traversal of the",
      "constructed AVL tree is")
myTree.preOrder(root)
print()


# In[33]:


#Heapsort
 
def heapify(arr, N, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
 
    # See if left child of root exists and is
    # greater than root
    if l < N and arr[largest] < arr[l]:
        largest = l
 
    # See if right child of root exists and is
    # greater than root
    if r < N and arr[largest] < arr[r]:
        largest = r
 
    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap
 
        # Heapify the root.
        heapify(arr, N, largest)
 
def heapSort(arr):
    N = len(arr)
 
    # Build a maxheap.
    for i in range(N//2 - 1, -1, -1):
        heapify(arr, N, i)
 
    # One by one extract elements
    for i in range(N-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)
 
 
# Driver's code
if __name__ == '__main__':
    arr = [12, 11, 13, 5, 6, 7]
 
    # Function call
    heapSort(arr)
    N = len(arr)
 
    print("Sorted array is")
    for i in range(N):
        print("%d" % arr[i], end=" ")


# In[34]:


#Prims
# A Python3 program for Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix representation of the graph

import sys # Library for INT_MAX


class Graph():

	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)]
					for row in range(vertices)]

	# A utility function to print the constructed MST stored in parent[]
	def printMST(self, parent):
		print("Edge \tWeight")
		for i in range(1, self.V):
			print(parent[i], "-", i, "\t", self.graph[i][parent[i]])

	# A utility function to find the vertex with
	# minimum distance value, from the set of vertices
	# not yet included in shortest path tree
	def minKey(self, key, mstSet):

		# Initialize min value
		min = sys.maxsize

		for v in range(self.V):
			if key[v] < min and mstSet[v] == False:
				min = key[v]
				min_index = v

		return min_index

	# Function to construct and print MST for a graph
	# represented using adjacency matrix representation
	def primMST(self):

		# Key values used to pick minimum weight edge in cut
		key = [sys.maxsize] * self.V
		parent = [None] * self.V # Array to store constructed MST
		# Make key 0 so that this vertex is picked as first vertex
		key[0] = 0
		mstSet = [False] * self.V

		parent[0] = -1 # First node is always the root of

		for cout in range(self.V):

			# Pick the minimum distance vertex from
			# the set of vertices not yet processed.
			# u is always equal to src in first iteration
			u = self.minKey(key, mstSet)

			# Put the minimum distance vertex in
			# the shortest path tree
			mstSet[u] = True

			# Update dist value of the adjacent vertices
			# of the picked vertex only if the current
			# distance is greater than new distance and
			# the vertex in not in the shortest path tree
			for v in range(self.V):

				# graph[u][v] is non zero only for adjacent vertices of m
				# mstSet[v] is false for vertices not yet included in MST
				# Update the key only if graph[u][v] is smaller than key[v]
				if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
					key[v] = self.graph[u][v]
					parent[v] = u

		self.printMST(parent)


# Driver's code
if __name__ == '__main__':
	g = Graph(5)
	g.graph = [[0, 2, 0, 6, 0],
			[2, 0, 3, 8, 5],
			[0, 3, 0, 0, 7],
			[6, 8, 0, 0, 9],
			[0, 5, 7, 9, 0]]

	g.primMST()


# In[35]:


# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

# Class to represent a graph


class Graph:

	def __init__(self, vertices):
		self.V = vertices # No. of vertices
		self.graph = []
		# to store graph

	# function to add an edge to graph
	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])

	# A utility function to find set of an element i
	# (truly uses path compression technique)
	def find(self, parent, i):
		if parent[i] != i:
		# Reassignment of node's parent to root node as
		# path compression requires
			parent[i] = self.find(parent, parent[i])
		return parent[i]

	# A function that does union of two sets of x and y
	# (uses union by rank)
	def union(self, parent, rank, x, y):
		
		# Attach smaller rank tree under root of
		# high rank tree (Union by Rank)
		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x

		# If ranks are same, then make one as root
		# and increment its rank by one
		else:
			parent[y] = x
			rank[x] += 1

	# The main function to construct MST using Kruskal's
		# algorithm
	def KruskalMST(self):

		result = [] # This will store the resultant MST

		# An index variable, used for sorted edges
		i = 0

		# An index variable, used for result[]
		e = 0

		# Step 1: Sort all the edges in
		# non-decreasing order of their
		# weight. If we are not allowed to change the
		# given graph, we can create a copy of graph
		self.graph = sorted(self.graph,
							key=lambda item: item[2])

		parent = []
		rank = []

		# Create V subsets with single elements
		for node in range(self.V):
			parent.append(node)
			rank.append(0)

		# Number of edges to be taken is equal to V-1
		while e < self.V - 1:

			# Step 2: Pick the smallest edge and increment
			# the index for next iteration
			u, v, w = self.graph[i]
			i = i + 1
			x = self.find(parent, u)
			y = self.find(parent, v)

			# If including this edge doesn't
			# cause cycle, then include it in result
			# and increment the index of result
			# for next edge
			if x != y:
				e = e + 1
				result.append([u, v, w])
				self.union(parent, rank, x, y)
			# Else discard the edge

		minimumCost = 0
		print("Edges in the constructed MST")
		for u, v, weight in result:
			minimumCost += weight
			print("%d -- %d == %d" % (u, v, weight))
		print("Minimum Spanning Tree", minimumCost)


# Driver's code
if __name__ == '__main__':
	g = Graph(4)
	g.addEdge(0, 1, 10)
	g.addEdge(0, 2, 6)
	g.addEdge(0, 3, 5)
	g.addEdge(1, 3, 15)
	g.addEdge(2, 3, 4)

	# Function call
	g.KruskalMST()

# This code is contributed by Neelam Yadav
# Improved by James Graça-Jones


# In[37]:


#Travelling salesman
from sys import maxsize
v = 4

def tsp(graph, source): 
    vertex = []
    for i in range(v): 
        if i != source:
            vertex.append(i)

    mininum_path = maxsize 
    while True:
        current_cost = 0
        k = source;
        for i in range(len(vertex)):
            current_cost += graph[k][vertex[i]]
            k = vertex[i]
        current_cost += graph[k][source]
        mininum_path = min(mininum_path, current_cost)

        if not nxt(vertex):
            break
        return mininum_path

def nxt(l):
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

g = [[0,10,15,17], [10,5,16,23], [15,35,5,23], [21,13,30,5]]
source = 0
result = tsp(g,source)

print("The cost of the cycle in TSP problem=",result)


# In[38]:


from collections import defaultdict
 
# This class represents a directed graph
# using adjacency list representation
class Graph:
 
    def __init__(self, vertices):
         
        # No. of vertices
        self.V = vertices
         
        # Default dictionary to store graph
        self.graph = defaultdict(list)
 
    # Function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
 
    # The function to print vertex cover
    def printVertexCover(self):
         
        # Initialize all vertices as not visited.
        visited = [False] * (self.V)
         
        # Consider all edges one by one
        for u in range(self.V):
             
            # An edge is only picked when
            # both visited[u] and visited[v]
            # are false
            if not visited[u]:
                 
                # Go through all adjacents of u and
                # pick the first not yet visited
                # vertex (We are basically picking
                # an edge (u, v) from remaining edges.
                for v in self.graph[u]:
                    if not visited[v]:
                         
                        # Add the vertices (u, v) to the
                        # result set. We make the vertex
                        # u and v visited so that all
                        # edges from/to them would
                        # be ignored
                        visited[v] = True
                        visited[u] = True
                        break
 
        # Print the vertex cover
        for j in range(self.V):
            if visited[j]:
                print(j, end = ' ')
                 
        print()
 
# Driver code
 
# Create a graph given in
# the above diagram
g = Graph(7)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 3)
g.addEdge(3, 4)
g.addEdge(4, 5)
g.addEdge(5, 6)
 
g.printVertexCover()


# In[ ]:




