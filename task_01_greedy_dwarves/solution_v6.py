import sys, math, time

class Agent:
    def __init__(self,K,N,T,W,m,v,mode=6):
        self.K,self.N,self.T,self.W=K,N,T,W
        self.m,self.v=m,v
        self.mode=mode
        self.r=0; self.M=[0]*K; self.V=0
        self.available=(1<<N)-1
        keys=[lambda i:v[i],lambda i:v[i]/m[i],lambda i:-m[i],lambda i:v[i]/math.sqrt(m[i]),lambda i:v[i]-2*m[i],lambda i:v[i]-1.5*m[i],lambda i:v[i]-3*m[i],lambda i:v[i]/(m[i]+5)]
        self.orders=[sorted(range(N),key=k,reverse=True) for k in keys]
        self.bits=[[0]*N for _ in keys]; self.caps=[[0]*(W+1) for _ in keys]
        for p,order in enumerate(self.orders):
            for rank,i in enumerate(order):
                b=1<<rank; self.bits[p][i]=b
                if m[i]<=W:self.caps[p][m[i]]|=b
            for c in range(1,W+1):self.caps[p][c]|=self.caps[p][c-1]
        self.rm=[self.available]*8; self.prob=[.125]*8
        self.ex=[math.exp(-.7*r) for r in range(N+1)]
        self.deadline=time.process_time()+7.2
        self.massmask=[0]*(W+1)
        for i in range(N):
            if m[i]<=W:self.massmask[m[i]]|=1<<i
        for c in range(1,W+1):self.massmask[c]|=self.massmask[c-1]

    def remove(self,i):
        self.available &= ~(1<<i)
        for p in range(8):self.rm[p]&=~self.bits[p][i]

    def commit(self,i):
        if i>=0:self.remove(i);self.M[0]+=self.m[i];self.V+=self.v[i]

    def observe(self,acts):
        for j,a in enumerate(acts,1):
            if not a:continue
            i=a-1;free=self.W-self.M[j]
            L=(self.rm[0]&self.caps[0][free]).bit_count()
            z=(1-self.ex[L])/(1-self.ex[1])
            for p in range(8):
                legal=self.rm[p]&self.caps[p][free]
                rank=(legal&(self.bits[p][i]-1)).bit_count()
                self.prob[p]*=.98*self.ex[rank]/z+.02/L
            s=sum(self.prob);self.prob=[x/s for x in self.prob] if s else [.125]*8
            self.remove(i);self.M[j]+=self.m[i]
        self.r+=1

    def rollout(self,first,bp,op,second=-1,safe=False):
        m,v,W,K=self.m,self.v,self.W,self.K
        bo,oo=self.orders[bp],self.orders[op];bb,ob=self.bits[bp],self.bits[op];bc,oc=self.caps[bp],self.caps[op]
        bm=self.rm[bp]&~bb[first];om=self.rm[op]&~ob[first]
        M=self.M.copy();M[0]+=m[first];value=self.V+v[first]
        if M[0]>=W:return 0
        for r in range(self.r,self.T):
            changed=False
            if r>self.r:
                c=W-M[0];legal=om&oc[c]
                if legal:
                    if safe and c>0 and legal&oc[c-1]:legal&=oc[c-1]
                    if r==self.r+1 and second>=0:
                        if not legal&ob[second]:return -1
                        i=second
                    else:i=oo[(legal&-legal).bit_length()-1]
                    M[0]+=m[i];value+=v[i];bm&=~bb[i];om&=~ob[i];changed=True
                    if M[0]>=W:return 0
                elif r==self.r+1 and second>=0:return -1
            for j in range(1,K):
                legal=bm&bc[W-M[j]]
                if legal:
                    i=bo[(legal&-legal).bit_length()-1];M[j]+=m[i];bm&=~bb[i];om&=~ob[i];changed=True
            if r>self.r and not changed:break
        return value if M[0]<max(M[1:]) else 0

    def next_state(self,first,bp):
        bm=self.rm[bp]&~self.bits[bp][first];av=self.available&~(1<<first)
        M=self.M.copy();M[0]+=self.m[first]
        for j in range(1,self.K):
            legal=bm&self.caps[bp][self.W-M[j]]
            if legal:
                i=self.orders[bp][(legal&-legal).bit_length()-1];bm&=~self.bits[bp][i];av&=~(1<<i);M[j]+=self.m[i]
        return av,M

    def choose(self):
        legal=self.available&self.massmask[self.W-self.M[0]]
        if not legal:return -1
        candidates=[]
        while legal:
            b=legal&-legal;candidates.append(b.bit_length()-1);legal^=b
        if len(candidates)==1:return candidates[0]
        scores=[]
        for i in candidates:
            score=0.;base=[]
            for bp in range(8):
                res=[self.rollout(i,bp,op) for op in (1,2,3)]
                s=.7*max(res)+.1*sum(res);base.append(max(res));score+=self.prob[bp]*s
            score+=.001*self.v[i]-.0001*self.m[i];scores.append((score,-i,i,base))
        scores.sort(reverse=True)
        if self.mode==5 or self.r==self.T-1 or time.process_time()>self.deadline:return scores[0][2]
        end=min(self.deadline,time.process_time()+.85*max(0.,self.deadline-time.process_time())/max(1,self.T-self.r))
        best_score=scores[0][0];best=scores[0][2]
        active=[];total=0.
        for p in sorted(range(8),key=self.prob.__getitem__,reverse=True):
            active.append(p);total+=self.prob[p]
            if total>=.97:break
        for score,_,i,base in scores[:8]:
            if time.process_time()>end:break
            if self.M[0]+self.m[i]>=self.W:continue
            gain=0.
            for bp in active:
                av,M=self.next_state(i,bp);c=self.W-M[0];eligible=av&self.massmask[c-1]
                newmax=max(base[bp],*(self.rollout(i,bp,op,safe=True) for op in (1,2,3)))
                options=[];tmp=eligible
                while tmp:
                    b=tmp&-tmp;options.append(b.bit_length()-1);tmp^=b
                if options:
                    top=set()
                    for op in (0,1,2,3,4):
                        hits=0
                        for j in self.orders[op]:
                            if eligible>>j&1:
                                top.add(j);hits+=1
                                if hits>=2:break
                    top.update(sorted(options,key=lambda j:(c-self.m[j],-self.v[j]))[:2])
                    if len(options)<=12:top.update(options)
                    for j in top:
                        for op in (1,2,3):
                            z=self.rollout(i,bp,op,second=j,safe=True)
                            if z>newmax:newmax=z
                gain+=self.prob[bp]*(newmax-base[bp])
            adjusted=score+.65*gain
            if adjusted>best_score:best_score=adjusted;best=i
        return best


def solve():
    read=sys.stdin.buffer.readline
    line=read()
    if not line:return
    K,N,T,W=map(int,line.split());m=list(map(int,read().split()));v=list(map(int,read().split()))
    a=Agent(K,N,T,W,m,v)
    for _ in range(T):
        i=a.choose();a.commit(i);print(i+1,flush=True)
        line=read()
        if not line:return
        acts=list(map(int,line.split()))
        if len(acts)!=K-1:return
        a.observe(acts)

if __name__=='__main__':solve()
