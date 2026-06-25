import torch
import sys
import numpy as np
import pandas as pd
import os
import time
import psutil
import matplotlib.pyplot as plt



def generate_data(N,D, gamma, noise,kappa_teacher=1.0):
    w = np.arange(1, int(D*kappa_teacher) + 1, dtype=np.float32)**(-gamma)
    if len(w)<D:
        w=np.concatenate((w,np.zeros(D-len(w))))
    #print(np.sqrt(D)/np.linalg.norm(w))
    w *= np.sqrt(D) / np.linalg.norm(w)
    trS = (w).sum()
    Q,_ = np.linalg.qr(np.random.randn(D, D))
    S_true = (Q * w) @ Q.T
    #print(f"Q0={np.sum(S_true**2)/D:.6e} ") #correct!

    x = np.random.randn(N, D).astype(np.float32)
    z = x  @ Q 
    quad = (z * z) @ w
    y = (quad - trS ) / np.sqrt(D) + np.sqrt(noise) * np.random.randn(N)
    #print(f"trS={trS:.6e} mean(y**2)={np.mean(y**2):.6e} signal={np.mean((quad-trS)**2)/D:.6e} diff={np.mean(y**2)- np.mean((quad-trS)**2)/D:.6e} noise={noise:.6e}")
    del z,Q,w
    return x,y,S_true

def experiment_single(x,y,ltilde,S_true,rank,options):
    dtype = torch.float32
    device = 'cpu'

    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    S_true = torch.as_tensor(S_true, dtype=dtype, device=device)
    trStrue = torch.trace(S_true).item()
    N, D = x.shape
    if rank is None:
        rank = D
    #print(f"Using rank={rank}")
    B = torch.randn(D, rank, dtype=dtype, device=device) / np.sqrt(D) 
    B.requires_grad_(True)

    D_t = torch.tensor(float(D), dtype=dtype, device=device)
    N_t = torch.tensor(float(N), dtype=dtype, device=device)
    sqrtD_t = torch.sqrt(D_t)
    it = {'k': 0, 'losses': [], 'regterm': [], 'trS': []}


    def closure():
        optimizer.zero_grad()
        Z = x @ B                 
        quad = (Z * Z).sum(dim=1)   
        trS = (B*B).sum() 
        yhat = (quad -trS) / sqrtD_t

        data = yhat - y
        regterm = ltilde *  trS/ D_t
        loss=(data @ data) / (D_t * D_t) + regterm
        loss.backward()
        it['k'] += 1
        it['losses'].append(loss.item())
        it['regterm'].append(regterm.item())
        it['trS'].append(trS.item())

        if options["verbose"] and it['k'] % options["print_every"] == 0:
            with torch.no_grad():
                S_hat = B @ B.T
                overlap_t = ((S_hat - S_true) ** 2).sum() / D_t
                print(f"iter {it['k']:4d}:  loss={loss.item():.6e}  trShat={trS.item():.6e}  trStrue={trStrue}  overlap={overlap_t.item():.6e}")
        return loss

    if options["optimizer"] == "LBFGS":
        optimizer = torch.optim.LBFGS([B], lr=1.0, max_iter=options["max_iter"],
                                    line_search_fn='strong_wolfe',
                                    tolerance_grad=options["tol"], tolerance_change=options["tol"]*0.01)

        optimizer.step(closure)
    elif options["optimizer"] == "GD":
        optimizer = torch.optim.SGD([B], lr=options["lr"],momentum=options["momentum"])

        for i in range(options["max_iter"]):
            optimizer.step(closure)
            if i>10 and abs(it['losses'][-2] - it['losses'][-1]) < options["tol"]:
                  break
            
    elif options["optimizer"] == "Adam":
        optimizer = torch.optim.Adam([B], lr=options["lr"])
        for i in range(options["max_iter"]):
            optimizer.step(closure)
            if i>10 and abs(it['losses'][-2] - it['losses'][-1]) < options["tol"]:
                break           
    
    

    if options["verbose"]:
        print(f"Converged to tolerance {options['tol']} after {it['k']} iterations, loss={it['losses'][-1]}")
             
    
    with torch.no_grad():
        S_hat = (B @ B.T)
        trS = torch.trace(S_hat).item()
        m=float(torch.sum(S_hat* S_true).cpu().item()/D)
        q=float(torch.sum(S_hat * S_hat).cpu().item() / D)
        overlap = float(((S_hat - S_true ) ** 2).sum().cpu().item() / D)
        eigs=torch.linalg.eigvalsh(S_hat.cpu().detach()).numpy()
        eigs=eigs[D-rank-1:]
        print(f"number of negative eigenvalues: {np.sum(eigs < 0)} out of {len(eigs)}")
        #print(f"iter {it['k']:4d}:  loss={it['losses'][-1]:.6e}  trShat={trS}  trStrue={trStrue}  overlap={overlap:.6e}")

    return overlap, it['losses'][-1],it['regterm'][-1],it['trS'][-1],it["k"],(q,m)



def experiment_avg(n,d,gamma,noise,ltilde,kappa_teacher,nsamples,rank,optimizer,verbose=False):
    mses = []
    losses = []
    ms = []
    qs = []
    regterms = []
    trSs = []
    for i in range(nsamples):
        x,y,S_true = generate_data(n, d, gamma, noise, kappa_teacher)

        if optimizer == "GD":
            options = {"optimizer": "GD", "lr": 1e-2, "momentum": 0.5, "max_iter": 150000, "tol": 1e-11,"verbose": verbose, "print_every": 100}
        elif optimizer == "LBFGS":
            options = {"optimizer": "LBFGS", "tol": 1e-10, "max_iter": 50000, "verbose": verbose, "print_every": 10}
        elif optimizer == "Adam":
            options = {"optimizer": "Adam", "lr": 1e-3, "max_iter": 50000, "tol": 1e-10, "verbose": verbose, "print_every": 100}

        mse,loss,regterm,trS,its,ovps=experiment_single(x,y,ltilde,S_true,rank,options=options)
        mses.append(mse)
        losses.append(loss)
        regterms.append(regterm)
        trSs.append(trS)
        ms.append(ovps[1])
        qs.append(ovps[0])
        print(f"Sample {i+1}/{nsamples}: mse={mse:.6e} loss={loss:.2e} regterm={regterm:.2e} trS={trS:.2e} trStrue={np.trace(S_true):.2e} iterations={its}")

    return np.mean(mses), np.std(mses), np.mean(losses),np.mean(regterms),np.mean(trSs),np.mean(ms),np.mean(qs)

def scan_ranks(alpha,d,ranks,ell,noise,gamma,kappa_teacher,nsamples,solver="LBFGS"):
    n=int(d**alpha)
    ltilde=d**ell
    df = pd.DataFrame( columns=['rank', 'd', 'mse', 'std','loss','regterm','trShat','ltilde','lreg','m','q'])
    folder=f"data_{solver}lowrank_ltilde/logd(n)={alpha:.3f}/logd(ltilde)_{ell:.2f}/"
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)

    filename=f"{folder}{solver}_ltilde_{ltilde:.3e}_n_{n}_d_{d}_gamma_{gamma:.2f}_kappa_star_{kappa_teacher:.1f}_noise_{noise}.csv"
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, header=True)
    else:
        df=pd.read_csv(filename)
        ranks_done=df['rank'].values
        ranks=[r for r in ranks if r not in ranks_done]
        print(f"{len(ranks_done)} Ranks already done: {ranks_done}, {len(ranks)} remaining ranks to do: {ranks}")
    del df
    verbose=False
    for i,r in enumerate(ranks):
        nsamples_r=nsamples if r>=10 else nsamples*2
        print(f"Running {solver} with rank={r}, ltilde={ltilde:.2e}",flush=True)
        mse,std,loss,regterm,trS,m,q=experiment_avg(n,d,gamma,noise,ltilde,kappa_teacher=kappa_teacher,nsamples=nsamples_r,rank=r,optimizer=solver,verbose=verbose)
    
        with open(filename, 'a') as f:
            f.write(f"{r},{d},{mse},{std},{loss},{regterm},{trS},{ltilde},{ltilde/np.sqrt(r/d)},{m},{q}\n")
        print(f"{solver} N={n}: mse={mse:.6e} +/- {std:.6e} with loss={loss:.2e} ")

  

def scan_n(d,rank,regstr,noise,gamma,kappa_teacher,nsamples,nvals,solver="LBFGS"):
    df = pd.DataFrame( columns=['n', 'd', 'mse', 'std','loss','ltilde','lreg','m','q'])
    kappa_stud=rank/d
    folder=f"data_{solver}lowrank_ltilde/kappastud={kappa_stud:.2f}/ltilde_{regstr}/"
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)
    filename=f"{folder}{solver}_kappa_{kappa_stud:.2f}_d_{d}_ltilde_{regstr}_kappa_star_{kappa_teacher:.1f}_noise_{noise}_gamma_{gamma:.2f}.csv"
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, header=True)
    else:
        df=pd.read_csv(filename)
        ranks_done=df['rank'].values
        ranks=[r for r in ranks if r not in ranks_done]
        print(f"{len(ranks_done)} Ranks already done: {ranks_done}, {len(ranks)} remaining ranks to do: {ranks}")
    del df
    ltilde=erm.parse_reg(regstr,n,d,gamma)
    verbose=False
    for i,n in enumerate(nvals):
        mse,std,loss,m,q,spectrum=experiment_avg(n,d,gamma,noise,ltilde,kappa_teacher=kappa_teacher,nsamples=nsamples,rank=rank,optimizer=solver,verbose=verbose)

        with open(filename, 'a') as f:
            f.write(f"{n},{d},{mse},{std},{loss},{ltilde},{ltilde/np.sqrt(kappa_stud)},{m},{q}\n")
        print(f"{solver} N={n}: mse={mse:.6e} +/- {std:.6e} with loss={loss:.2e} ")




def scan_n(d,alphas,ell,noise,gamma,rho_stud,nsamples,solver="LBFGS"):
    kappa_teacher=1.0
    ltilde=d**ell
    r=int(d**rho_stud)
    df = pd.DataFrame( columns=['rank', 'd', 'mse', 'std','loss','regterm','trShat','ltilde','lreg','m','q'])
    folder=f"data_{solver}lowrank_ltilde/rhostud={rho_stud:.3f}/logd(ltilde)_{ell:.2f}/"
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)

    filename=f"{folder}{solver}_logd(ltilde)_{ell:.2f}_rank_{r}_d_{d}_gamma_{gamma:.2f}_noise_{noise}.csv"
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, header=True)

    del df
    verbose=False
    for i,alpha in enumerate(alphas):
        n=int(d**alpha)
        print(f"Running {solver} with alpha={alpha}, n={n}, ltilde={ltilde:.2e}",flush=True)
        mse,std,loss,regterm,trS,m,q=experiment_avg(n,d,gamma,noise,ltilde,kappa_teacher=kappa_teacher,nsamples=nsamples,rank=r,optimizer=solver,verbose=verbose)
    
        with open(filename, 'a') as f:
            f.write(f"{r},{d},{mse},{std},{loss},{regterm},{trS},{ltilde},{ltilde/np.sqrt(r/d)},{m},{q}\n")
        print(f"{solver} N={n}: mse={mse:.6e} +/- {std:.6e} with loss={loss:.2e} ")


if __name__ == "__main__":

    d=100
    nsamples = 3
    #noise=float(sys.argv[2])
    kappa_teacher=1.0
    alphas=[2.5]
    
    solver="LBFGS"
    nvals=np.unique(np.logspace(3, 4, 10, dtype=int))
    ranks=np.unique(np.logspace(np.log10(100), np.log10(d), 5, dtype=int))
    gamma=0.6
    #rank=int(sys.argv[3])
    for noise in [0.05]:
        for alpha in alphas: #[(0.1*d)**(2*gamma)/d]:
            for ell in [alpha/2-0.9]: 
                print(f"Starting d={d} alpha={alpha} gamma={gamma}  noise={noise}")
                start=time.time()
                scan_ranks(alpha,d, ranks,ell, noise, gamma,kappa_teacher,nsamples,solver=solver)
                #scan_n(d,rank,regstr,noise,gamma,kappa_teacher,nsamples,nvals,solver="LBFGS",save_spectra=False)
                end=time.time()
                print(f"===============Finished {solver} scan_ranks in {end-start:.2f} seconds================")
    print("All done!")
