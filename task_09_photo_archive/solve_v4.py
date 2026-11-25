#!/usr/bin/env python3
"""Task 9 V4: crop-level distance-signature refinement of V2 clusters."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse, csv
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from solve import read_csv, make_global_descriptor

cv2.setNumThreads(1)

def visual_similarity(scene_dir, rows):
    hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    gd=[]; cd=[]; thumbs=[]
    for row in rows:
        im=cv2.cvtColor(cv2.imread(str(scene_dir/row["image_path"])),cv2.COLOR_BGR2RGB)
        mask=cv2.imread(str(scene_dir/row["mask_path"]),0)
        gd.append(make_global_descriptor(im,mask,hog))
        lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB)
        px=lab[mask>0]
        if not len(px): px=lab.reshape(-1,3)
        hist=np.histogramdd(px,bins=(12,8,8),range=((0,256),)*3)[0].ravel()
        cd.append(np.sqrt(hist/(hist.sum()+1e-9)))
        im[mask==0]=127
        thumbs.append(cv2.resize(im,(12,12),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255.)
    gd=np.asarray(gd,dtype=np.float32)
    cd=np.asarray(cd,dtype=np.float32)
    thumbs=np.asarray(thumbs,dtype=np.float32)
    sim=.45*(gd@gd.T)+.35*(cd@cd.T)+.20*(1.-np.linalg.norm(thumbs[:,None,:]-thumbs[None,:,:],axis=2)/np.sqrt(thumbs.shape[1]))
    return np.clip(sim,0.,1.)

def process_scene(scene_dir, base_labels):
    rows=read_csv(scene_dir/"crops.csv")
    drows=read_csv(scene_dir/"distances.csv")
    labels=np.asarray([base_labels[r["crop_id"]] for r in rows],dtype=np.int32)
    sim=visual_similarity(scene_dir,rows)
    n=len(rows)
    idx={r["crop_id"]:i for i,r in enumerate(rows)}
    adj=[{} for _ in range(n)]
    forbidden_pairs=set()
    for r in drows:
        i,j=idx[r["crop_id_a"]],idx[r["crop_id_b"]]
        dd=float(r["distance_m"])
        adj[i][j]=dd; adj[j][i]=dd
        a,b=int(labels[i]),int(labels[j])
        if a!=b: forbidden_pairs.add((min(a,b),max(a,b)))

    sig=[]
    for i in range(n):
        by_group=defaultdict(list)
        for j,dd in adj[i].items():
            g=int(labels[j])
            if g!=int(labels[i]): by_group[g].append(dd)
        sig.append({g:float(np.median(v)) for g,v in by_group.items()})

    members={int(g):np.flatnonzero(labels==g) for g in np.unique(labels)}
    evidence=defaultdict(list)
    for i in range(n):
        gi=int(labels[i]); si=sig[i]
        for j in range(i+1,n):
            gj=int(labels[j])
            if gi==gj: continue
            visual=float(sim[i,j])
            if visual<.55: continue
            common=si.keys() & sig[j].keys()
            if len(common)<2: continue
            diffs=[abs(si[g]-sig[j][g]) for g in common]
            med=float(np.median(diffs)); worst=float(max(diffs))
            if med>=.30 or worst>=.90: continue
            a,b=(gi,gj) if gi<gj else (gj,gi)
            if (a,b) in forbidden_pairs: continue
            if gi==a: evidence[(a,b)].append((i,j,med,visual,len(common)))
            else: evidence[(a,b)].append((j,i,med,visual,len(common)))

    candidates=[]
    for (a,b),ev in evidence.items():
        aa,bb=members[a],members[b]
        posa={int(x):k for k,x in enumerate(aa)}
        posb={int(x):k for k,x in enumerate(bb)}
        rr=[]; cc=[]
        for i,j,med,vis,k in ev:
            if i in posa and j in posb:
                rr.append(posa[i]); cc.append(posb[j])
        if not rr: continue
        mat=csr_matrix((np.ones(len(rr),dtype=np.int8),(rr,cc)),shape=(len(aa),len(bb)))
        matching=maximum_bipartite_matching(mat,perm_type="column")
        match_count=int(np.sum(matching>=0))
        small=min(len(aa),len(bb))
        coverage=match_count/max(1,small)
        meds=np.asarray([x[2] for x in ev])
        visuals=np.asarray([x[3] for x in ev])
        anchors=np.asarray([x[4] for x in ev])
        med_med=float(np.median(meds))
        max_med=float(np.max(meds))
        mean_vis=float(np.mean(visuals))
        best_vis=float(np.max(visuals))
        mean_anchors=float(np.mean(anchors))

        if small<=2:
            ok=(match_count>=2 and coverage>=1.0 and len(ev)>=2 and
                med_med<.23 and max_med<.50 and mean_vis>.62 and best_vis>.68)
        elif small<=5:
            ok=(match_count>=2 and coverage>=.60 and len(ev)>=2 and
                med_med<.24 and max_med<.55 and mean_vis>.62 and best_vis>.68)
        elif small<=15:
            ok=(match_count>=3 and coverage>=.60 and len(ev)>=8 and
                med_med<.23 and max_med<.65 and mean_vis>.60 and best_vis>.68)
        else:
            ok=(match_count>=8 and coverage>=.90 and len(ev)>=8 and
                med_med<.16 and max_med<.45 and mean_vis>.60 and best_vis>.70)
        if not ok: continue

        rank=(2.*coverage+.035*min(match_count,12)+.012*min(len(ev),30)
              +.35*mean_vis+.10*best_vis+.025*min(mean_anchors,6)
              -1.2*med_med-.35*max_med)
        candidates.append((rank,a,b,len(ev),match_count,coverage,med_med,max_med,mean_vis,best_vis))

    groups=sorted(members)
    parent={g:g for g in groups}
    component={g:{g} for g in groups}
    forbidden={g:set() for g in groups}
    for a,b in forbidden_pairs:
        forbidden[a].add(b); forbidden[b].add(a)

    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x

    accepted=[]
    for cand in sorted(candidates,reverse=True):
        rank,a0,b0=cand[:3]
        a,b=find(a0),find(b0)
        if a==b: continue
        conflict=False
        for old in component[a]:
            if forbidden[old].intersection(component[b]):
                conflict=True; break
        if conflict: continue
        if len(component[a])<len(component[b]): a,b=b,a
        parent[b]=a
        component[a].update(component.pop(b))
        accepted.append(cand)

    roots=[find(int(x)) for x in labels]
    remap={}; refined=[]
    for x in roots:
        if x not in remap: remap[x]=len(remap)
        refined.append(remap[x])
    refined=np.asarray(refined,dtype=np.int32)

    for r in drows:
        if refined[idx[r["crop_id_a"]]]==refined[idx[r["crop_id_b"]]]:
            raise RuntimeError(f"{scene_dir.name}: cannot-link violation")

    out=[(scene_dir.name,row["crop_id"],int(refined[i])) for i,row in enumerate(rows)]
    stats={"scene":scene_dir.name,"crops":n,"v2_groups":len(members),
           "v4_groups":len(np.unique(refined)),"candidate_merges":len(candidates),
           "accepted_merges":len(accepted),
           "singletons":int(np.sum(np.bincount(refined)==1)),
           "cannot_link_violations":0}
    return out,stats,accepted

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True,help="Directory with submission_v2.csv")
    ap.add_argument("--output",type=Path,default=Path("submission_v4.csv"))
    ap.add_argument("--stats",type=Path,default=Path("scene_stats_v4.csv"))
    ap.add_argument("--merges",type=Path,default=Path("merge_candidates_v4.csv"))
    args=ap.parse_args()

    base_rows=read_csv(args.base/"submission_v2.csv")
    all_base=defaultdict(dict)
    for r in base_rows:
        all_base[r["scene_id"]][r["crop_id"]]=int(r["object_id"])

    all_out=[]; all_stats=[]; all_merges=[]
    for scene in sorted(p for p in args.data.iterdir() if p.is_dir() and p.name.startswith("scene_")):
        out,stats,merges=process_scene(scene,all_base[scene.name])
        all_out.extend(out); all_stats.append(stats)
        for m in merges:
            rank,a,b,support,matching,cov,med,maxmed,mvis,bvis=m
            all_merges.append((scene.name,a,b,rank,support,matching,cov,med,maxmed,mvis,bvis))
        print(f'{scene.name}: {stats["v2_groups"]}->{stats["v4_groups"]}; '
              f'merges={stats["accepted_merges"]}; cannot-link=0',flush=True)

    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["scene_id","crop_id","object_id"]); w.writerows(all_out)
    with args.stats.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_stats[0]))
        w.writeheader(); w.writerows(all_stats)
    with args.merges.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["scene","group_a","group_b","rank","support","matching","coverage",
                    "median_profile_diff","max_pair_median_diff","mean_visual","best_visual"])
        w.writerows(all_merges)
    print("Total crops:",len(all_out))
    print("Total groups:",sum(x["v4_groups"] for x in all_stats))
    print("Accepted merges:",sum(x["accepted_merges"] for x in all_stats))
    print("Cannot-link violations:",sum(x["cannot_link_violations"] for x in all_stats))

if __name__=="__main__":
    main()
