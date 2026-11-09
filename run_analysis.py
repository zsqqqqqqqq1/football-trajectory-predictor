"""Run reproducible contest example, posterior propagation, texture/profile checks."""
import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from model import Physics, fit, propagate, sample_posterior, drag_coefficient, DRAG_NODES, PARAMETER_NAMES

ROOT=Path(__file__).resolve().parent
R0=np.array([0.,0.,.11]); TIMES=np.array([.05,.10])
OBS=np.array([[1.50,.45,.44],[2.96,.86,.74]])


def clean(x):
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    return x


def save_json(path,data):
    Path(path).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(pred,statuses):
    ok=np.array(statuses)=='crossed'; p=pred[ok]; yz=p[:,1:]
    out=dict(n=len(pred),n_crossed=int(ok.sum()),p_cross=float(ok.mean()),
             p_ground=float(np.mean(np.array(statuses)=='ground')),
             p_timeout=float(np.mean(np.array(statuses)=='timeout')))
    if 'past_target' in statuses:
        out['p_past_target'] = float(np.mean(np.array(statuses) == 'past_target'))
    inside=(np.abs(yz[:,0])<3.66)&(yz[:,1]>0)&(yz[:,1]<2.44)
    out.update(p_goal=float(inside.sum() / len(pred)),
               p_goal_given_cross=float(inside.mean()) if len(p) else None)
    if len(p) < 20 or np.linalg.matrix_rank(yz - yz.mean(axis=0)) < 2:
        return out, None
    kde=gaussian_kde(yz.T)
    std=yz.std(axis=0); low=yz.min(axis=0)-.6*std; high=yz.max(axis=0)+.6*std
    gy=np.linspace(low[0],high[0],220);gz=np.linspace(low[1],high[1],220)
    Y,Z=np.meshgrid(gy,gz);density=kde(np.vstack([Y.ravel(),Z.ravel()])).reshape(Y.shape)
    area=(gy[1]-gy[0])*(gz[1]-gz[0]);ordered=np.sort(density.ravel())[::-1]
    total=float(density.sum()*area);level=float(ordered[np.searchsorted(np.cumsum(ordered)*area,.95*total)])
    idx=np.unravel_index(np.argmax(density),density.shape); mode0=np.array([Y[idx],Z[idx]])
    mode=minimize(lambda p:-float(kde(np.asarray(p).reshape(2,1))[0]),mode0,method='Nelder-Mead').x
    mask=(np.abs(Y)<3.66)&(Z>0)&(Z<2.44)
    idxc=np.unravel_index(np.argmax(np.where(mask,density,-1)),density.shape)
    conditional_mode=None
    if inside.sum()>=20:
        # Same goal-plane KDE restricted and renormalized to the goal rectangle.
        opt=minimize(lambda p:-float(kde(np.asarray(p).reshape(2,1))[0]),[Y[idxc],Z[idxc]],
                     method='L-BFGS-B',bounds=[(-3.66+1e-9,3.66-1e-9),(1e-9,2.44-1e-9)])
        conditional_mode=opt.x
    out.update(mean_tyz=p.mean(axis=0),median_tyz=np.median(p,axis=0),
               marginal_95_tyz=np.quantile(p,[.025,.975],axis=0),mode_yz=mode,
               conditional_goal_mode_yz=conditional_mode,
               region_95_area=float(np.sum(density>=level)*area),kde_level_95=level,
               kde_grid_mass=total,covariance_yz=np.cov(yz.T),
               region_definition='95% KDE highest-density region conditional on crossing before ground; numerical grid',
               probability_definition='geometric center strictly inside given rectangle; denominator includes all samples')
    return out,dict(Y=Y,Z=Z,density=density,level=level)


def posterior_job(args):
    noise,prior,draws,burn,seed,outdir=args; outdir=Path(outdir)
    result=sample_posterior(R0,TIMES,OBS,noise=noise,prior=prior,draws=draws,burn=burn,seed=seed)
    ss=result['samples']; events=[propagate(t,R0) for t in ss]
    statuses=[e['status'] for e in events]; pred=np.array([[e['tg'],e['yg'],e['zg']] for e in events])
    summary,grid=summarize(pred,statuses)
    summary.update(noise=noise,prior=prior,diagnostics=result['diagnostics'],map_theta=result['fit']['theta'],
                   map_event=propagate(result['fit']['theta'],R0),prior_spec=result['fit']['prior'],
                   parameter_names=PARAMETER_NAMES, physics=asdict(Physics()))
    name=f'{noise}_{prior}'
    np.savez_compressed(outdir/f'{name}_samples.npz',parameters=ss,parameter_names=PARAMETER_NAMES,chains=result['chains'],tyz=pred,statuses=statuses)
    if grid is not None: np.savez_compressed(outdir/f'{name}_density.npz',**grid)
    save_json(outdir/f'{name}_summary.json',summary)
    return summary


def deterministic(outdir):
    rows=[];fits={}
    for name,kw in [('gravity',dict(model='gravity',regularize=False)),('drag',dict(model='drag',regularize=False)),
                    ('full_unregularized',dict(regularize=False)),('full_MAP_wide',{}),
                    ('full_MAP_strong',dict(prior='strong')),('fixed_CD_022',dict(fixed_cd=.22,regularize=False))]:
        f=fit(R0,TIMES,OBS,**kw); e=propagate(f['theta'],R0,fixed_cd=f['fixed_cd']);fits[name]=f
        rows.append(dict(name=name,**e,cd_initial=(f['fixed_cd'] if f['fixed_cd'] is not None else
                                    drag_coefficient(np.linalg.norm(f['theta'][:3]))), speed_initial=np.linalg.norm(f['theta'][:3]),
                         max_residual_mm=1000*np.max(np.abs(f['residual_m'])),condition_number=f['condition_number']))
    save_csv(outdir/'point_estimates.csv',rows)
    save_json(outdir/'identifiability.json',{k:{x:v for x,v in f.items() if x not in ('jac',)} for k,f in fits.items()})
    texture=[]
    for cd in [.15,.18,.22]:
        f=fit(R0,TIMES,OBS,fixed_cd=cd,regularize=True);e=propagate(f['theta'],R0,fixed_cd=f['fixed_cd'])
        main=next(row for row in rows if row['name']=='full_MAP_wide')
        texture.append(dict(cd=cd,**e,max_residual_mm=1000*np.abs(f['residual_m']).max(),
                            rms_residual_mm=1000*np.sqrt(np.mean(f['residual_m']**2)),
                            delta_tg=e['tg']-main['tg'],delta_yg=e['yg']-main['yg'],delta_zg=e['zg']-main['zg']))
        if cd==.15: fits['texture_low']=f
    save_csv(outdir/'texture_sensitivity.csv',texture)
    profile=[]
    for cd in np.linspace(.05,.65,25):
        f=fit(R0,TIMES,OBS,fixed_cd=cd,regularize=False)
        profile.append(dict(cd=cd,chi2_gaussian_1cm=float(np.sum((f['residual_m']/.01)**2)),
                            max_residual_mm=float(1000*np.abs(f['residual_m']).max()),**propagate(f['theta'],R0,fixed_cd=f['fixed_cd'])))
    save_csv(outdir/'cd_profile.csv',profile)
    cases=[]
    for name in ['gravity','full_MAP_wide','texture_low']:
        f=fits[name];theta=f['theta']
        physics = Physics()
        cases.append(dict(name=name, r0=R0, v0=theta[:3], omega0=theta[3:], cd=f['fixed_cd'] or 0.,
                          use_fixed_cd=f['fixed_cd'] is not None,
                          cd_nodes=DRAG_NODES[physics.drag_profile], wind=[0, 0, 0],
                          constants=asdict(physics), target_x=11,
                          python=propagate(theta, R0, fixed_cd=f['fixed_cd'])))
    save_json(outdir/'matlab_cases.json',dict(cases=cases,python_rtol=1e-10,python_atol=1e-12))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--draws',type=int,default=4000);p.add_argument('--burn',type=int,default=1500)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--deterministic-only',action='store_true');a=p.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True);deterministic(out)
    if a.deterministic_only: return
    jobs=[(n,p,a.draws,a.burn,20261107+i,str(out)) for i,(n,p) in enumerate(
        [('gaussian','wide'),('gaussian','strong'),('rounding','wide'),('rounding','strong')])]
    with ProcessPoolExecutor(max_workers=a.workers) as pool: summaries=list(pool.map(posterior_job,jobs))
    save_json(out/'posterior_summary.json',summaries)
    for s in summaries: print(s['noise'],s['prior'],clean({k:s.get(k) for k in ['mode_yz','p_goal','region_95_area','diagnostics']}),flush=True)


if __name__=='__main__': main()
