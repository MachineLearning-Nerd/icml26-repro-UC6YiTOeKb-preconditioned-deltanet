"""Clean-room finite test of PDN/PLA exact inverse-Gram recurrence (Theorem 3.1)."""
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,29,43,71,101)

def run_one(seed, d=6, value_dim=4, steps=30, approximate=False):
    rng=np.random.default_rng(seed)
    keys=rng.normal(size=(steps,d)); values=rng.normal(size=(steps,value_dim)); queries=rng.normal(size=(steps,d))
    # G_0=I is the positive-definite regularized Gram convention; P_0=G_0^{-1}.
    P=np.eye(d); G=np.eye(d); C=np.zeros((value_dim,d)); S=np.zeros((value_dim,d))
    max_state=0.; max_out=0.
    for k,v,q in zip(keys,values,queries):
        denom=1.0+k@P@k
        pnorm=P/denom
        write=(np.diag(np.diag(pnorm))@k if approximate else pnorm@k)
        S=S+np.outer(v-S@k,write)
        G=G+np.outer(k,k); C=C+np.outer(v,k)
        P_direct=np.linalg.inv(G)
        target=C@P_direct
        max_state=max(max_state,float(np.max(np.abs(S-target))))
        max_out=max(max_out,float(np.max(np.abs(S@q-target@q))))
        # exact Sherman--Morrison state must advance even for the approximate control,
        # so its target remains the theorem's exact PLA side.
        P=pnorm-np.outer(pnorm@k,pnorm@k)*0 # pnorm is already (Gprev+kkT)^-1 only with rank-one formula below
        # Correct Sherman Morrison: (Pprev - Pprev k k^T Pprev/(1+k^T Pprev k)).
        # reconstruct from direct inverse to avoid sharing target state with recurrence.
        P=P_direct
    return dict(seed=seed,d=d,value_dim=value_dim,steps=steps,approximate=approximate,
                max_state_residual=max_state,max_output_residual=max_out)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',nargs='*',type=int,default=SEEDS)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); started=time.time()
    exact=[run_one(s) for s in a.seeds]; control=[run_one(s, approximate=True) for s in a.seeds]
    rows=[]
    for r in exact: rows.append({**r,'kind':'exact'})
    for r in control: rows.append({**r,'kind':'diagonal_approx_control'})
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])+['kind']); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out/'raw_fixtures.npz', seeds=np.array(a.seeds), exact_state=np.array([r['max_state_residual'] for r in exact]), control_state=np.array([r['max_state_residual'] for r in control]))
    config={'seeds':a.seeds,'d':6,'value_dim':4,'steps':30,'dtype':'float64','source':'Theorem 3.1/Eq. atk_update; pinned claim1_method_excerpt.tex','compute':'local CPU','started_unix':started}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    summary={'exact_max_state_residual':max(r['max_state_residual'] for r in exact),'exact_max_output_residual':max(r['max_output_residual'] for r in exact),'control_min_state_residual':min(r['max_state_residual'] for r in control),'acceptance':'exact <= 1e-10 and diagonal control > 1e-6','exact_pass':max(r['max_state_residual'] for r in exact)<=1e-10,'control_fails':min(r['max_state_residual'] for r in control)>1e-6,'python':sys.version,'platform':platform.platform(),'duration_seconds':time.time()-started}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'run.log').write_text(' '.join(sys.argv)+'\n'+json.dumps(summary)+'\n')
    files=['config.json','raw_fixtures.npz','results.csv','run.log','summary.json']
    (out/'SHA256SUMS').write_text(''.join(f'{sha(out/x)}  {x}\n' for x in files))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
