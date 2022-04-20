import time
from queue import PriorityQueue
import random
import math

class HArc:
    def __init__(self, id, u, v, base, num, den, mul, low) -> None:
        self.id = id
        self.u = u
        self.v = v
        self.base = base
        self.num = num
        self.den = den
        self.mul = mul
        self.low = low
        
    def contains(self, b):
        return self.u<=b.u and b.v<=self.v
    
    def get_support(self):
        assert(self.den!=0)
        return self.num/self.den
    
    def __lt__(self, b):
        return self.get_support() < b.get_support()
    
    def __le__(self, b):
        return self.get_support() <= b.get_support()
    
    def __eq__(self, b):
        return self.get_support() == b.get_support()




def empty(lst):
    return len(lst)==0

def new_arc(u, v):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    assert(u <= v)

    n_arcs += 1
    h[n_arcs].id = n_arcs
    h[n_arcs].u = u
    h[n_arcs].v = v
    h[n_arcs].low = u if w[u]<w[v] else v
    h[n_arcs].mul = w[u]*w[v]
    h[n_arcs].base = CP[v]-CP[u]-h[n_arcs].mul

def build_tree(lst):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    stk = []
    new_arc(1, n+1)
 
    for it in lst:
        new_arc(it[0], it[1])
 
        while not empty(stk) and h[n_arcs].contains(h[stk[-1]]):
            child[n_arcs].append(stk.pop(-1))
            
        stk.append(n_arcs)
    while not empty(stk):
        child[1].append(stk.pop(-1))
    
def one_sweep():
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    stk = []
    tmp = []
    lst = []

    
    for i in range(1, n+1):
        
        while len(stk) >= 2 and w[stk[-1]] > w[i]:
            tmp.append([stk[len(stk)-2], i])
            stk.pop(-1)
        stk.append(i)
    
    while len(stk)>=4:
        Vt_1 = stk[len(stk)-2]
        tmp.append([1, Vt_1])
        stk.pop(-1)
    
    for it in tmp:
        if it[0] == 1 or it[1] == 1:
            continue
        lst.append(it)
    
    build_tree(lst)
    

def prepare():
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con

    V1 = w.index(min(w[1:n+1]))

    tempw = [x for x in w if x>0]
    tempw = tempw[V1-1:] + tempw[:V1-1]
    w[1:n+1] = tempw

    w[n+1] = w[1]
    for i in range(1, n+2):
        CP[i] = w[i]*w[i-1]
        CP[i] += CP[i-1]
    
def get_mn_mul(node):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    if node==1:
        return w[1]*w[2]+w[1]*w[n]
    cur = h[node]
    if cur.u == cur.low:
        if empty(con[cur.u]) or not cur.contains(con[cur.u][-1]):
            return w[cur.u]*w[cur.u+1]
        else:
            return con[cur.u][-1].mul
    else:
        if empty(con[cur.v]) or not cur.contains(con[cur.v][-1]):
            return w[cur.v]*w[cur.v - 1]
        else:
            return con[cur.v][-1].mul
    
def add_arc(cur_node, arc):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con

    pq[qid[cur_node]].put(arc)

    # print(pq[qid[cur_node]])
        
    con[arc.u].append(arc)
    con[arc.v].append(arc)

def remove_arc(cur_node):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    
    # print(qid[cur_node])
    
    hm = pq[qid[cur_node]].get()
    # print(hm.u)
    # print()

    
    con[hm.u].pop(-1)
    con[hm.v].pop(-1)
    

def merge_pq(node):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    max_child = -1
    for it in child[node]:
        if max_child == -1 or sub[max_child]<sub[it]:
            max_child = it
    qid[node] = qid[max_child]
    cur_pq = pq[qid[node]]
    for it in child[node]:
        if it == max_child:
            continue
        child_pq = pq[qid[it]]
        while not child_pq.empty():
            cur_pq.put(child_pq.get())
            

def dfs(node):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    
    cur = h[node]
    sub[node] = 1
    if empty(child[node]):
        n_pqs += 1
        qid[node] = n_pqs
        cur.den = cur.base
        cur.num = w[cur.low]*(cur.den + cur.mul - get_mn_mul(node))
        add_arc(node, cur)
        return
    
    cur.den = cur.base
    
    for it in child[node]:
        dfs(it)
        sub[node] += sub[it]
        cur.den -= h[it].base
    
    cur.num = w[cur.low]*(cur.den + cur.mul - get_mn_mul(node))
    merge_pq(node)
    cur_pq = pq[qid[node]]

    while not cur_pq.empty() and cur_pq.queue[0].get_support() >= w[cur.low]:
        hm = cur_pq.queue[0]
        cur.den += hm.den
        remove_arc(node)
        cur.num = w[cur.low]*(cur.den + cur.mul - get_mn_mul(node))

   
    while not cur_pq.empty() and cur <= cur_pq.queue[0]:
        hm = cur_pq.queue[0]
        cur.den += hm.den
        remove_arc(node)
        cur.num += hm.num

    add_arc(node, cur)

def getans():
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    dfs(1)
    ans = 0
    cur_pq = pq[qid[1]]
    while not cur_pq.empty():
        ans += cur_pq.get().num
        
    
    return ans

def input_(arr):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    n = len(arr)
    for i in range(1, n+1):
        w[i] = arr[i-1]

def solve(arr):
    global n, n_arcs, n_pqs, h, w, CP, sub, child, qid, pq, con
    if len(arr) < 2:
        return 0
    if len(arr) == 2:
        return arr[0]*arr[1]
    input_(arr)
    

    prepare()
    one_sweep()

    return getans()

maxn = 1200
n = 0
n_arcs = 0
n_pqs = 0
h = [HArc(0, 0, 0, 0, 0, 0, 0, 0) for _ in range(maxn)]
w = [0 for _ in range(maxn)]
CP = [0 for _ in range(maxn)]
sub = [0 for _ in range(maxn)]
child = [[] for _ in range(maxn)]
qid = [0 for _ in range(maxn)]
pq = [PriorityQueue() for _ in range(maxn)]
con = [[] for _ in range(maxn)]


# arr = random.sample(range(5, 1000), 500)
# arr = random.sample(range(5, 1000), 700)
arr = random.sample(range(5, 1000), 850)


# print(arr)

       
start_time=time.time()

print("Minimum scalar operations will be: ", solve(arr))

end_time=time.time()

print("Execution time for Hu-Shing", end_time-start_time)



# initialize maxn x maxn memo table
memo = [[-1 for i in range(maxn)] for j in range(maxn)]


def mcm(k, i, j):
    if memo[i][j] != -1:
        return memo[i][j]
    if i == j:
        return 0
    memo[i][j] = math.inf
    for l in range(i, j):
        memo[i][j] = min(
            memo[i][j],
            mcm(k, i, l) + mcm(k, l+1, j) + k[j] * k[l] * k[i-1]
        )
    return memo[i][j]




start_time=time.time()

n_dp = len(arr)
i = 1
j = n_dp - 1
answer = mcm(arr, i, j)
print("Minimum scalar operations will be:", answer)

end_time=time.time()

print("Execution time for DP: ", end_time-start_time)