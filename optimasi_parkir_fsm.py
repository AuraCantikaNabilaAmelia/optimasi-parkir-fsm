"""
Optimasi Pencarian Slot Parkir - FSM UNDIP
Implementasi Lengkap & Terpadu: Pencarian (UCS, GBFS, A*, DP, Bellman-Ford) & Visualisasi
"""
import heapq
import math
import time
from collections import defaultdict, deque

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# 1. Konfigurasi Parameter
ALPHA = 1.5          # Koefisien sensitivitas kemacetan
SLOT_EDGE = 4.0      # Jarak (m) dari simpang ke slot
GRID_STEP = 15.0     # Jarak antar simpang (m)
ROWS, COLS = 4, 4    # Grid 4x4 simpang

# 2. Pembangunan GRAF & Skenario
def build_graph():
    nodes = {}
    slots = {}
    
    # Gerbang masuk
    nodes['G'] = (-GRID_STEP, 0.0)
    for i in range(ROWS):
        for j in range(COLS):
            nodes[f'N{i}{j}'] = (j*GRID_STEP, i*GRID_STEP)
            
    # Slot parkir (2 slot per simpang)
    sid = 1
    slot_anchor = {}
    for i in range(ROWS):
        for j in range(COLS):
            for dy, tag in [(SLOT_EDGE, 'a'), (-SLOT_EDGE, 'b')]:
                x, y = nodes[f'N{i}{j}']
                s = f'P{i}{j}{tag}'
                slots[s] = (x, y+dy)
                slot_anchor[s] = f'N{i}{j}'
                sid += 1
    return nodes, slots, slot_anchor

NODES, SLOTS, SLOT_ANCHOR = build_graph()

def dist(a_xy, b_xy):
    return math.hypot(a_xy[0]-b_xy[0], a_xy[1]-b_xy[1])

def base_edges():
    E = []
    # Gerbang -> N00
    E.append(('G', 'N00', dist(NODES['G'], NODES['N00'])))
    for i in range(ROWS):
        for j in range(COLS):
            if j+1 < COLS:
                E.append((f'N{i}{j}', f'N{i}{j+1}', GRID_STEP))
            if i+1 < ROWS:
                E.append((f'N{i}{j}', f'N{i+1}{j}', GRID_STEP))
    for s, anc in SLOT_ANCHOR.items():
        E.append((anc, s, SLOT_EDGE))
    return E

BASE_E = base_edges()

def density_of_edge(u, v, level):
    pu = NODES.get(u, SLOTS.get(u))
    pv = NODES.get(v, SLOTS.get(v))
    my = (pu[1]+pv[1])/2
    base = {'rendah': 0.10, 'sedang': 0.40, 'tinggi': 0.75}[level]
    prox = max(0.0, 1.0 - my/(ROWS*GRID_STEP))
    rho = min(1.0, base + 0.25*prox*{'rendah':0.5, 'sedang':1.0, 'tinggi':1.2}[level])
    return rho

def occupied_slots(level):
    ratio = {'rendah': 0.30, 'sedang': 0.55, 'tinggi': 0.80}[level]
    names = sorted(SLOTS.keys())
    occ = set()
    for idx, s in enumerate(names):
        y = SLOTS[s][1]
        score = (idx*37 % 100)/100.0 - 0.30*(1 - y/(ROWS*GRID_STEP))
        if score < ratio:
            occ.add(s)
    return occ

def build_weighted(level, directed_dag=False):
    adj = defaultdict(list)
    occ = occupied_slots(level)
    
    def key(n):  
        p = NODES.get(n, SLOTS.get(n))
        return p[0]+p[1]
        
    for u, v, d in BASE_E:
        rho = density_of_edge(u, v, level)
        w = d * (1 + ALPHA * rho)
        if directed_dag:
            if u in SLOTS or v in SLOTS:
                a, b = (u, v) if v in SLOTS else (v, u)
            else:
                a, b = (u, v) if key(u) <= key(v) else (v, u)
            adj[a].append((b, w))
            # Pastikan node tujuan terdaftar di adj agar Kahn's algorithm aman
            if b not in adj: adj[b] = []
        else:
            adj[u].append((v, w))
            adj[v].append((u, w))
            
    available = [s for s in SLOTS if s not in occ]
    return adj, available, occ

def make_heuristic(available):
    pts = [SLOTS[s] for s in available]
    def h(n):
        p = NODES.get(n, SLOTS.get(n))
        return min(dist(p, q) for q in pts) if pts else 0.0
    return h

def reconstruct(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return path[::-1]

# 3. Algoritma Pencarian
def ucs(adj, start, goals):
    goals = set(goals); pq = [(0.0, start)]; g = {start:0.0}; parent={start:None}
    expanded=0; maxfront=1; visited=set()
    while pq:
        maxfront = max(maxfront, len(pq))
        c, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); expanded += 1
        if u in goals: return c, reconstruct(parent, u), expanded, maxfront
        for v, w in adj[u]:
            ng = c + w
            if ng < g.get(v, float('inf')):
                g[v] = ng; parent[v] = u; heapq.heappush(pq, (ng, v))
    return None, None, expanded, maxfront

def greedy_bfs(adj, start, goals, h):
    goals = set(goals); pq = [(h(start), start)]; parent = {start:None}; gcost = {start:0.0}
    expanded=0; maxfront=1; visited=set()
    while pq:
        maxfront = max(maxfront, len(pq))
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); expanded += 1
        if u in goals: return gcost[u], reconstruct(parent, u), expanded, maxfront
        for v, w in adj[u]:
            if v not in visited:
                gcost[v] = gcost[u] + w; parent[v] = u; heapq.heappush(pq, (h(v), v))
    return None, None, expanded, maxfront

def astar(adj, start, goals, h):
    goals = set(goals); pq = [(h(start), 0.0, start)]; g = {start:0.0}; parent = {start:None}
    expanded=0; maxfront=1; visited=set()
    while pq:
        maxfront = max(maxfront, len(pq))
        _, c, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); expanded += 1
        if u in goals: return c, reconstruct(parent, u), expanded, maxfront
        for v, w in adj[u]:
            ng = c + w
            if ng < g.get(v, float('inf')):
                g[v] = ng; parent[v] = u; heapq.heappush(pq, (ng + h(v), ng, v))
    return None, None, expanded, maxfront

def topo_order(adj_dag):
    indeg = defaultdict(int)
    nodes = set(adj_dag.keys())
    for u in adj_dag:
        for v, _ in adj_dag[u]:
            indeg[v] += 1
            nodes.add(v)
            
    q = [n for n in nodes if indeg[n] == 0]
    order = []
    dq = deque(q)
    
    while dq:
        u = dq.popleft()
        order.append(u)
        for v, _ in adj_dag[u]:
            indeg[v] -= 1
            if indeg[v] == 0: 
                dq.append(v)
    return order

def dp_dag(adj_dag, start, goals):
    order = topo_order(adj_dag)
    g = {n: float('inf') for n in order}; g[start] = 0.0; parent = {start:None}
    relax = 0
    
    for u in order:
        if g[u] == float('inf'): continue
        for v, w in adj_dag[u]:
            relax += 1
            if g[u] + w < g[v]:
                g[v] = g[u] + w; parent[v] = u
                
    best, bg = None, float('inf')
    for t in goals:
        if g.get(t, float('inf')) < bg: 
            bg = g[t]; best = t
            
    if best is None: return None, None, relax, len(order)
    return bg, reconstruct(parent, best), relax, len(order)

def bellman_ford(adj, start, goals):
    nodes = set(adj.keys())
    for u in adj:
        for v, _ in adj[u]: nodes.add(v)
    edges = [(u, v, w) for u in adj for v, w in adj[u]]
    
    g = {n: float('inf') for n in nodes}; g[start] = 0.0; parent = {start:None}
    relax = 0
    
    for _ in range(len(nodes)-1):
        changed = False
        for u, v, w in edges:
            relax += 1
            if g[u] + w < g[v]:
                g[v] = g[u] + w; parent[v] = u; changed = True
        if not changed: break
        
    best, bg = None, float('inf')
    for t in goals:
        if g.get(t, float('inf')) < bg: 
            bg = g[t]; best = t
            
    if best is None: return None, None, relax, len(nodes)
    return bg, reconstruct(parent, best), relax, len(nodes)

# 4. Eksperimen & Visualisasi
def time_call(fn, repeats=100):
    t0 = time.perf_counter()
    for _ in range(repeats): r = fn()
    dt = (time.perf_counter() - t0) / repeats * 1000.0  
    return r, dt

def run_experiments():
    results = {}
    for level in ['rendah', 'sedang', 'tinggi']:
        adj, avail, occ = build_weighted(level, directed_dag=False)
        adj_dag, avail2, _ = build_weighted(level, directed_dag=True)
        h = make_heuristic(avail)
        start = 'G'
        row = {}
        
        (r, dt) = time_call(lambda: ucs(adj, start, avail)); row['UCS'] = (*r, dt)
        (r, dt) = time_call(lambda: greedy_bfs(adj, start, avail, h)); row['Greedy BFS'] = (*r, dt)
        (r, dt) = time_call(lambda: astar(adj, start, avail, h)); row['A*'] = (*r, dt)
        (r, dt) = time_call(lambda: dp_dag(adj_dag, start, avail)); row['DP (DAG)'] = (*r, dt)
        (r, dt) = time_call(lambda: bellman_ford(adj, start, avail)); row['Bellman-Ford'] = (*r, dt)
        
        results[level] = {'avail': len(avail), 'occ': len(occ), 'algos': {}}
        for k, (cost, path, ops, mem, dt) in row.items():
            results[level]['algos'][k] = {
                'cost': round(cost, 2) if cost else None,
                'goal': path[-1] if path else None,
                'len': len(path) if path else None,
                'ops': ops, 'mem': mem, 'time_ms': round(dt, 4),
                'path': path
            }
    return results

def draw_visualizations(res):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.grid':True,
                         'grid.alpha':0.3, 'figure.dpi':120})
    C = {'UCS':'#1f4e79', 'Greedy BFS':'#c0392b', 'A*':'#27ae60', 'DP (DAG)':'#8e44ad', 'Bellman-Ford':'#e67e22'}

    # Gambar 1: GRAF (Sedang)
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.6))
    occ = occupied_slots('sedang')
    for u, v, d in BASE_E:
        pu = NODES.get(u, SLOTS.get(u)); pv = NODES.get(v, SLOTS.get(v))
        ax1.plot([pu[0], pv[0]], [pu[1], pv[1]], color='#b0b0b0', lw=1, zorder=1)
    for n, (x, y) in NODES.items():
        if n == 'G':
            ax1.scatter([x], [y], c='#000000', s=160, marker='s', zorder=3)
            ax1.annotate('Gerbang', (x, y), textcoords='offset points', xytext=(-6, -14), fontsize=8, ha='center')
        else:
            ax1.scatter([x], [y], c='#34495e', s=40, marker='o', zorder=3)
    for s, (x, y) in SLOTS.items():
        if s in occ: ax1.scatter([x], [y], c='#c0392b', s=28, marker='x', zorder=3)
        else: ax1.scatter([x], [y], c='#27ae60', s=34, marker='o', edgecolors='#145a32', zorder=3)
    ax1.set_aspect('equal'); ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    
    leg = [Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=10, label='Gerbang masuk'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#34495e', markersize=8, label='Simpang jalur'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=8, label='Slot tersedia'),
           Line2D([0], [0], marker='x', color='#c0392b', markersize=8, label='Slot terisi', lw=0)]
    ax1.legend(handles=leg, loc='upper left', fontsize=8, framealpha=0.9)
    ax1.set_title('Model Graf Area Parkir FSM (Skenario Sedang)', fontsize=10)
    plt.tight_layout()

    # Gambar 2: Lintasan Rendah
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.6))
    occ_rendah = occupied_slots('rendah')
    for u, v, d in BASE_E:
        pu = NODES.get(u, SLOTS.get(u)); pv = NODES.get(v, SLOTS.get(v))
        ax2.plot([pu[0], pv[0]], [pu[1], pv[1]], color='#d5d5d5', lw=1, zorder=1)
    for n, (x, y) in NODES.items():
        ax2.scatter([x], [y], c=('k' if n == 'G' else '#7f8c8d'), s=(120 if n == 'G' else 22), marker=('s' if n == 'G' else 'o'), zorder=2)
    for s, (x, y) in SLOTS.items():
        ax2.scatter([x], [y], c=('#c0392b' if s in occ_rendah else '#27ae60'), s=22, marker=('x' if s in occ_rendah else 'o'), zorder=2)
        
    def plot_path(path, color, label, off):
        xs = [(NODES.get(p, SLOTS.get(p)))[0] + off for p in path]
        ys = [(NODES.get(p, SLOTS.get(p)))[1] + off for p in path]
        ax2.plot(xs, ys, color=color, lw=2.4, label=label, zorder=4, alpha=0.9)
        
    plot_path(res['rendah']['algos']['A*']['path'], C['A*'], f"A* (Optimal, Biaya {res['rendah']['algos']['A*']['cost']})", 0.4)
    plot_path(res['rendah']['algos']['Greedy BFS']['path'], C['Greedy BFS'], f"Greedy BFS (Biaya {res['rendah']['algos']['Greedy BFS']['cost']})", -0.4)
    
    ax2.set_aspect('equal'); ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.set_title('Perbandingan Lintasan: Optimal (A*) vs Greedy BFS (Skenario Rendah)', fontsize=10)
    plt.tight_layout()

    # Gambar 3: Metrik
    algos = ['UCS', 'Greedy BFS', 'A*', 'DP (DAG)', 'Bellman-Ford']
    levels = ['rendah', 'sedang', 'tinggi']
    
    fig3, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    x = np.arange(len(levels)); w = 0.16
    metrics_config = [('cost', axes[0], 'Biaya Lintasan Optimal', 'Biaya total'),
                      ('ops', axes[1], 'Jumlah Operasi', 'Ekspansi simpul / relaksasi'),
                      ('time_ms', axes[2], 'Waktu Eksekusi', 'Waktu (ms)')]
                      
    for met, ax, ttl, yl in metrics_config:
        d = {a: [res[l]['algos'][a][met] for l in levels] for a in algos}
        for i, a in enumerate(algos):
            ax.bar(x + (i - 2) * w, d[a], w, label=a, color=C[a])
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in levels])
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_xlabel('Skenario kepadatan', fontsize=9)
        if met == 'ops': ax.set_yscale('log')
        
    axes[1].legend(fontsize=8, loc='upper left', ncol=1)
    plt.tight_layout()
    
    # Menampilkan semua plot secara interaktif
    plt.show()

# MAIN
if __name__ == '__main__':
    print("Mengeksekusi simulasi pencarian lintasan parkir...")
    res = run_experiments()
    
    # Cetak ringkasan ke terminal
    for lvl, d in res.items():
        print(f"\n=== Kepadatan {lvl.upper()} | Slot tersedia: {d['avail']}, Terisi: {d['occ']} ===")
        print(f"{'Algoritma':<14}{'Biaya':>8}{'Slot':>8}{'Ops':>7}{'Mem':>6}{'Waktu(ms)':>11}")
        print("-" * 56)
        for k, v in d['algos'].items():
            cost_str = str(v['cost']) if v['cost'] else "N/A"
            goal_str = str(v['goal']) if v['goal'] else "N/A"
            print(f"{k:<14}{cost_str:>8}{goal_str:>8}{v['ops']:>7}{v['mem']:>6}{v['time_ms']:>11.4f}")
            
    print("\nSimulasi selesai. Membangun visualisasi Matplotlib...")
    draw_visualizations(res)