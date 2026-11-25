#!/usr/bin/env python3
"""Mars settlement agent — Python 3.11, protocol v3."""
from __future__ import annotations
import sys
from dataclasses import dataclass

def read_line():
    s = sys.stdin.readline()
    if s == "":
        raise EOFError("unexpected EOF")
    return s.rstrip("\r\n")

def dec(s: str, d: int) -> int:
    a, _, b = s.partition(".")
    if len(b) > d:
        raise ValueError("decimal precision")
    return int(a) * (10 ** d) + int((b + "0" * d)[:d] or "0")

@dataclass
class Game:
    w:int; h:int; n:int; t:int; d:int; scale:int; a:int; b:int
    stations:list[tuple[int,int]]; blocked:set[tuple[int,int]]

def read_initial():
    w,h=map(int,read_line().split())
    n=int(read_line()); t=int(read_line())
    r0=read_line(); aa,bb=read_line().split()
    d=max(len(x.partition(".")[2]) for x in (r0,aa,bb))
    k=int(read_line())
    blocked={tuple(map(int,read_line().split())) for _ in range(k)}
    stations=[tuple(map(int,read_line().split())) for _ in range(n)]
    g=Game(w,h,n,t,d,10**d,dec(aa,d),dec(bb,d),stations,blocked|set(stations))
    return g,[dec(r0,d)]*n,[{} for _ in range(n)]

def neighbors(g, c):
    x,y=c
    for q in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
        if 0<=q[0]<g.w and 0<=q[1]<g.h:
            yield q

def accessible(g, robots, p):
    out=set()
    roots=[g.stations[p]]
    roots.extend(c for c,v in robots[p].items() if v)
    for c in roots:
        for q in neighbors(g,c):
            if q not in g.blocked:
                out.add(q)
    return out

def income(g, own, enemy):
    total=own+enemy
    if total == 0: return 0
    return own*max(0,g.a-g.b*total)

def choose(g, turn, resources, robots):
    left=g.t-turn+1
    budget=resources[0]//g.scale
    if budget<=0: return {}
    cells=accessible(g,robots,0)
    if not cells: return {}
    enemy_total={}
    for p in range(1,g.n):
        for c,v in robots[p].items():
            enemy_total[c]=enemy_total.get(c,0)+v
    own_now={c:robots[0].get(c,0) for c in cells}
    req={}
    spent=0
    while spent < budget:
        best=None
        for c in cells:
            x=own_now[c]
            e=enemy_total.get(c,0)
            delta=income(g,x+1,e)-income(g,x,e)
            if delta<=0:
                continue
            bonus=0
            if x==0 and left>1:
                fresh=sum(1 for q in neighbors(g,c)
                          if q not in g.blocked and q not in cells
                          and q not in g.stations)
                bonus=int(fresh*(left-1)*(g.a-g.b)*0.12)
            value=delta*left+bonus
            if value>g.scale and (best is None or value>best[0]):
                best=(value,c)
        if best is None:
            break
        _,c=best
        own_now[c]+=1
        req[c]=req.get(c,0)+1
        spent+=1
    return req

def parse_request(line):
    tok=line.split()
    if len(tok)%3: raise ValueError("bad request")
    ans={}
    for i in range(0,len(tok),3):
        c=(int(tok[i]),int(tok[i+1])); v=int(tok[i+2])
        if v<=0 or c in ans: raise ValueError("bad triple")
        ans[c]=v
    return ans

def apply(g, robots, requests):
    avail=[accessible(g,robots,p) for p in range(g.n)]
    for p,req in enumerate(requests):
        for c,v in req.items():
            if c in avail[p]:
                robots[p][c]=robots[p].get(c,0)+v

def main():
    g,res,robots=read_initial()
    for turn in range(1,g.t+1):
        req=choose(g,turn,res,robots)
        line=" ".join(f"{x} {y} {v}" for (x,y),v in sorted(req.items()))
        sys.stdout.write(line+"\n"); sys.stdout.flush()
        requests=[req]
        for _ in range(g.n-1):
            requests.append(parse_request(read_line()))
        rr=read_line().split()
        if len(rr)!=g.n: raise ValueError("bad resources")
        res=[dec(x,g.d) for x in rr]
        apply(g,robots,requests)

if __name__=="__main__":
    main()
