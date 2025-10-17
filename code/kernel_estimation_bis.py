from tv_deconv import tv_deconv
import numpy as np 
import math
import scipy
import scipy.linalg as la

cos, sin, tan, pi, atan2 = np.cos, np.sin, np.tan, np.pi, np.arctan2

fft2 = scipy.fft.rfft2
ifft2 = scipy.fft.irfft2
real = np.real
conj = np.conj


def get_phase_alea(Mh,s=None):
    if s is None:
        s=Mh
    im=np.random.randn(Mh,Mh)
    #im[s:,:]=0
    #im[:,s:]=0
    return np.angle(np.fft.fft2(im))

def from_module_phase(module, phase):
    return np.real(np.fft.ifft2(module * np.exp(1j * phase)))

def SinglePhaseRetrieval(module,s,Mh=32,alpha=0.95,beta0=0.75,Ninner=300,known=None):
    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    phg=get_phase_alea(Mh,s)
    modg=module

    if known is None:
        known=np.ones((Mh,Mh))>0
    unknown=(np.vectorize(lambda x: not x))(known)
    modg[unknown]=0
    g=from_module_phase(modg,phg)

    for m in range(Ninner):

        beta=beta0+(1-beta0)*(1-np.exp(-(m/7)**3))
        module_for_reconst=modg.copy()
        module_for_reconst[known]=(alpha*module+(1-alpha)*modg)[known]
        gp=from_module_phase(module_for_reconst,phg)
        mask=(2*gp<g)
        mask[s:,:]=True
        mask[:,s:]=True
        g[mask]=beta*g[mask]+(1-2*beta)*gp[mask]
        invmask=(np.vectorize(lambda x: not x))(mask)
        g[invmask]=gp[invmask]
        fg=fft2(g)
        phg=np.angle(fg)
        modg=abs(fg)
    g[g<0]=0
    g[s:,:]=0
    g[:,s:]=0
    g=g/g.sum()
    g[g<(1/255)]=0
    g/=g.sum()
    return g[:s,:s]

def Dx_Dy(im):
    """ Renvoie deux images de la même taille que im. im est une image en
    niveaux de gris.
    Ce sont les dérivées suivant x et suivant y en utilisant le noyau
    dérivateur spécial """
    d=np.asarray([3,-32,168,-672,0,672,-168,32,-3])/840
    d=d.reshape((1,-1))
    Dx=scipy.signal.convolve2d(im, d,mode='same',boundary='symm')
    Dy=scipy.signal.convolve2d(im, d.T,mode='same',boundary='symm')
    return (Dx,Dy)


def entre_Mpi2_pi2(x):
    pi=np.pi
    y=x%pi
    if y>=pi/2:
        y-=pi
    return y

def liste_thetas_depuis_spectre(N):
    """ Renvoies la liste des thetas telles que pour tout point du
    spectre discret de taille NxN corresponde un angle"""
    if N%2==0:
        tmp=np.concatenate((np.arange(0,N//2+1),np.arange(-N//2+1,0)))
    else:
        tmp=np.concatenate((np.arange(0,(N+1)//2),np.arange(-(N-1)//2,0)))
    tmp=tmp.astype(int)
    X,Y=np.meshgrid(tmp,tmp) # carte des fréquences
    pi=np.pi
    Xs=X.reshape(-1)
    Ys=Y.reshape(-1)
    lt=[]
    c=0
    for k in range(len(Xs)):
        if Xs[k]>=0 and np.gcd(Xs[k],Ys[k])==1:
            c+=1
            an=math.atan2(Ys[k],Xs[k])
            lt.append(entre_Mpi2_pi2(an))
    lt=np.asarray(list(set(lt))) #unique
    lt=np.sort(lt)
    surech=0*lt
    for k in range(len(lt)):
        an=lt[k]
        if abs(an)<pi/4:
            surech[k]=1/cos(abs(an))
        else:
            surech[k]=1/sin(abs(an))
    return lt,surech

def projections_rapide_shear(tab,thetas,demitaille):
    """ projette tab sur toutes les droites d'angles dans thetas.
    La sortie est centree autour de la projection du point central de l'image.
    Nous faisons la projection shear: c'est à dire pas la projection orthogonale
    Cela respecte la méthode originale dans l'article
    Les angles sont censés être entre -pi/2 et pi/2
    Mais leur ordre importe peu.
    """

    L=len(thetas)
    cs=cos(thetas)
    ss=sin(thetas)
    ccs=cs.copy()
    css=ss.copy()
    # modification pour tenir compte du shear
    for m in range(len(thetas)):
        t=(thetas[m])
        if abs(t)<=pi/4:
            ccs[m]=1
            css[m]=tan(t)
        else:
            css[m]=1
            ccs[m]=cos(t)/sin(t)


    M,N=tab.shape
    T=M+N+4
    #le pixel central tombe toujours au même endroit
    poscentre=(T)//2
    kc=M//2
    lc=N//2
    out=np.zeros((L,T))
    outs1=out.reshape(-1)
    vals=np.zeros(L)
    Ls=(T*np.arange(0,L)).astype(int)
    for k in range(M):
        print(k/M*100,'%',end='\r')
        for l in range(N):
            pos=np.round(ccs*(l-lc)+css*(k-kc)).astype(int)+poscentre
            if max(pos)>T:
                print('offenders',k,l)
            vals[:]=tab[k,l]
            outs1[Ls+pos]+=vals
    return (out[:,poscentre-demitaille:poscentre+demitaille+1])

def projections_rapide_gradient_shear(DDx,DDy,DDxy,DDyx,thetas,demitaille):
    """ utile les intercorralations de Dx,Dy avec eux même pour calculer
    les projections des autocorrelations du noyau
    """

    L=len(thetas)
    cs=cos(thetas)
    ss=sin(thetas)
    cs2=cs**2
    ss2=ss**2
    csss=cs*ss
    ccs=cs.copy()
    css=ss.copy()
    # modification pour tenir compte du shear
    for m in range(len(thetas)):
        t=thetas[m]
        if abs(t)<=pi/4:
            ccs[m]=1
            css[m]=tan(t)
        else:
            css[m]=1
            ccs[m]=cos(t)/sin(t)


    M,N=DDx.shape
    T=M+N+4
    #le pixel central tombe toujours au même endroit
    poscentre=(T)//2
    kc=M//2
    lc=N//2

    out=np.zeros((L,T))
    outs1=out.reshape(-1)
    vals=np.zeros(L)
    Ls=(T*np.arange(0,L)).astype(int)
    for k in range(M):
        print(k/M*100,'%',end='\r')
        for l in range(N):
            pos=np.round(ccs*(l-lc)+css*(k-kc)).astype(int)+poscentre
            if max(pos)>T-1:
                print('offenders',k,l)
            #print("debug", k,l,DDx.shape)
            vals[:]=cs2*DDx[k,l]+ss2*DDy[k,l]+csss*(DDxy[k,l]+DDyx[k,l])
            #if vals[0]!=DDy[k,l]:
            #    print('Houston we have a problem',k,l,vals[0],DDy[k,l],cs2[0],ss2[0],csss[0])
            outs1[Ls+pos]+=vals
    return (out[:,poscentre-demitaille:poscentre+demitaille+1])

def next_power_2(T):# renvoie la puissance de deux immédiatement supérieure
    return int(2**(np.floor(np.log(T)/np.log(2))+1))

def decoupe(X,A): #decoupe une partie de tableau et la fftshift
    out1=np.concatenate((X[:A+1,-A:],X[:A+1,:A+1]),axis=1)
    out2=np.concatenate((X[-A:,-A:],X[-A:,:A+1]),axis=1)
    out=np.concatenate((out2,out1),axis=0)
    return out
def calcul_correlations_initiales(img,thetas,p):
    """ estime l'autocorrelation du noyau à partir du gradient de l'image
    La deconvolution et le filtrage median sont fait ailleurs"""



    (Dx,Dy)=Dx_Dy(img)
    # calcul des trois correlations
    (a,b)=Dx.shape
    A=next_power_2(2*a)
    B=next_power_2(2*b)
    fDx=fft2(Dx,(A,B))
    fDy=fft2(Dy,(A,B))
    DDx=real(ifft2(abs(fDx)**2))
    DDy=real(ifft2(abs(fDy)**2))
    DDxy= real(ifft2(fDx*conj(fDy)))
    DDyx= real(ifft2(fDy*conj(fDx)))
    DDx=decoupe(DDx,2*p)
    DDy=decoupe(DDy,2*p)
    DDxy=decoupe(DDxy,2*p)
    DDyx=decoupe(DDyx,2*p)


    out=projections_rapide_gradient_shear(DDx,DDy,DDxy,DDyx,thetas,2*p)
    return out


def deconv_intrinsic_blur(corr, alpha=2.1, lam=1e-2):
    _, qp1 = corr.shape
    M = np.zeros((qp1, qp1))
    for k in range(qp1):
        for l in range(qp1):
            M[k,l] = 1.0 / ((abs(k-l)+1)**alpha)
    M /= M[0,:].sum()
    # résoudre (M + λI) x = y, plus stable que inv(M) @ y
    A = M + lam*np.eye(qp1)
    deconv = la.solve(A, corr.T, assume_a='pos', check_finite=False).T

    poscentre = qp1//2
    bad = (deconv[:, poscentre-2:poscentre+2].min(axis=1) < 0)
    deconv[bad] = corr[bad]           # fallback comme avant
    deconv[~bad] = np.maximum(deconv[~bad], 0)  # coupe les petits négatifs résiduels
    return deconv


def initial_support_estimation(tab_corrs,centre,thetas,kappa=30):
    tab_interet=tab_corrs[:,centre:]
    sprime=tab_interet.argmin(axis=1)
    s=(tab_interet.shape[1]-1)*np.ones(tab_interet.shape[0])
    for k in range(tab_interet.shape[0]):
        if sprime[k]<s[k]:
            s[k]=sprime[k]
            for m in range(tab_interet.shape[0]):
                s[m]=min(s[m],\
                                   sprime[k]+\
                                       kappa*abs(thetas[m]-thetas[k]))
    return s

def Estimate_h_correlations(tab_corrs,supports):
    """ si le support est connu, on met à zéro tout ce qui dépasse.
       on enleve R[s] à tout le monde
       on normalise à somme 1"""
    centre=tab_corrs.shape[1]//2
    new_corrs=tab_corrs.copy()
    for k in range(new_corrs.shape[0]):
        sint=int(np.round(supports[k]))
        new_corrs[k,:]-=new_corrs[k,centre+sint]
        new_corrs[k,:centre-sint+1]=0
        new_corrs[k,centre+sint:]=0
        new_corrs[new_corrs<0]=0
        new_corrs[k,:]/=(new_corrs[k,:]).sum()
    # filtrage median circulaire
    taille=int(np.ceil(new_corrs.shape[0]**0.5))
    out=np.zeros(new_corrs.shape)
    #taille=0 # supprier le filtrage
    for k in range(new_corrs.shape[0]):
        if k-taille<0:
            tabmed=np.concatenate((new_corrs[:k+taille,:],new_corrs[k-taille:,:]),axis=0)
        elif k+taille+1>new_corrs.shape[0]:
            tabmed=np.concatenate((new_corrs[k-taille:,:],\
                                   new_corrs[:((k+taille+1)%new_corrs.shape[0]),:]),axis=0)
        else:
            tabmed=new_corrs[k-taille:k+taille+1,:]
        out[k,:]=np.median(tabmed,axis=0)
    return out


def Restimation_supports_noyau(h,p,thetas,ratio=0.05):
    """recalcule les autocorrelations
    du noyau a partir d'une nouvelle estiation et recalcule les supports"""

    (a,b)=h.shape
    A=next_power_2(2*a+1)
    B=next_power_2(2*b+1)
    fh=fft2(h,(A,B))
    autocor=real(ifft2(abs(fh)**2))
    autocor=decoupe(autocor,p)
    #affiche(autocor)

    proj=projections_rapide_shear(autocor,thetas,p)
    #affiche(proj)
    centre=p
    idxs=np.arange(p+1)
    out=np.zeros(thetas.shape[0])
    for k in range(proj.shape[0]):
        ma=proj[k].max()
        mask=proj[k,centre:]>(ratio*ma)
        out[k]=(idxs*mask).max()
    return out


def calcul_indices_passage_corr_power_spectrum_kernel(N,sc,thetas):
    """Calcule des indices tels que
     f=fft_corr[indices]
     donnera dans f le power spectrum de taille NxN en shape (-1)
     du noyau. fft_corr est la trasnformée de Fourier des correlations
     (en shape(-1)).
     sc est un couple
     thetas sont les angles choisis pour la correlation
     Ce calcul d'indices est fait une fois pour toute pour tous les calculs.
     N : fft2 du noyau doit etre de forme carree NxN
    """


    _,w=sc

    if N%2==0:
        tmp=np.concatenate((np.arange(0,N//2+1),np.arange(-N//2+1,0)))
    else:
        tmp=np.concatenate((np.arange(0,(N+1)//2),np.arange(-(N-1)//2,0)))
    [XX,YY]=np.meshgrid(tmp/N,tmp/N) # les frequences
    #angle=np.empty(XX.shape,dtype=np.float32)
    numligne=np.empty(XX.shape,dtype=int)
    posdansligne=np.empty(XX.shape,dtype=int)
    indexs=np.zeros(N*N,dtype=int)
    for k in range(XX.shape[0]):
        for l in range(XX.shape[1]):
            angle=entre_Mpi2_pi2( atan2(YY[k,l],XX[k,l]))
            #if angle<-pi/2:
            #    angle+=pi
            #elif angle>pi/2:
            #    angle-=pi
            numligne[k,l]=abs(thetas-angle).argmin()
            #if abs(thetas-angle).min()>0.001:
            #    print ('probleme')
            d=(XX[k,l]**2+YY[k,l]**2)**0.5
            if abs(thetas[numligne[k,l]])<pi/4:
                maxd=1/cos(abs(thetas[numligne[k,l]]))
            else:
                maxd=1/sin(abs(thetas[numligne[k,l]]))
            posdansligne[k,l]=min(int(np.round(d/maxd*w)),w//2)

            indexs[k*N+l]=numligne[k,l]*w+posdansligne[k,l]
    return indexs,numligne,posdansligne

def spectre_puissance_depuis_corrs(tab_corrs, N, indexs):
    """ Transforme des corrélations en spectre de puissance (NxN)
        en utilisant des indices précalculés : indexs doit pointer
        dans un tableau aplati de taille h*w.
    """
    h, w = tab_corrs.shape
    center = (w - 1) // 2

    # zero-padding / recentrage sur largeur w (PAS N)
    tmp = np.zeros((h, w))
    tmp[:, :center+1] = tab_corrs[:, center:]     # [centre..fin] → début
    tmp[:, -center:]  = tab_corrs[:, :center]     # [0..centre-1] → fin

    # FFT 1D sur l’axe des colonnes (longueur = w)
    fcorr = np.fft.fft(tmp, axis=1)               # shape = (h, w)

    out = np.zeros((N, N))

    # Aplats & sécurités
    fcorrs1 = fcorr.reshape(-1, order='C')        # taille = h*w
    outs1   = out.reshape(-1,   order='C')        # taille = N*N

    # Garde-fous utiles
    assert indexs.dtype.kind in 'iu', "indexs doit être entier"
    assert indexs.size == outs1.size, f"indexs.size={indexs.size} != N*N={N*N}"
    assert indexs.max() < fcorrs1.size, f"indexs.max()={indexs.max()} >= h*w={h*w}"

    outs1[:] = np.real(fcorrs1[indexs])
    outs1[outs1 < 0] = 0
    return out

def convol_carre(im,taille):
    """covole tres rapidement contre un carre"""
    im=im.cumsum(axis=0)
    im=im[taille:,:]-im[:-taille,:]
    im=im.cumsum(axis=1)
    im=im[:,taille:]-im[:,:-taille]
    return im

def calcul_variances_patchs(img,taille):
    """Renvoie une image des variances des patchs de taille=taille X taille
    La sortie de cette fonction permet de trouver dez petites zones dans
    l'image sur lesquels tester la deconvolution"""
    imgmoy=convol_carre(img,taille)/(taille**2)
    imgvar=convol_carre(img**2,taille)
    imgvar=imgvar-(imgmoy**2)*(taille**2)
    return imgvar

def Propose_patch_haute_variance(Varianceimage,img,taille):
    h,w=Varianceimage.shape
    xs=np.random.randint(0,high=w,size=10)
    ys=np.random.randint(0,high=h,size=10)
    pos=Varianceimage.reshape(-1)[xs+ys*w].argmax()
    print(xs[pos],ys[pos])
    return img[ys[pos]:ys[pos]+taille,xs[pos]:xs[pos]+taille].copy()

def score_restau(im): # utilisée pour comparer des resultats de deconvolution
    # entre eux.
    dx=im[:-1,1:]-im[:-1,:-1]
    dy=im[1:,:-1]-im[:-1,:-1]
    n=((dx**2)+(dy**2))**0.5
    return n.sum()/(((n**2).sum())**0.5)



def estime_noyau(img,p=25,Nouter=3,Ntries=30,Ninner=300,\
                 verbose=True):
    taille_patch=150
    imgvars=calcul_variances_patchs(img,taille_patch)
    Nspectrenoyau=4*p+1
    thetas,_=liste_thetas_depuis_spectre(Nspectrenoyau)
    Nthetas=thetas.shape[0]
    indexs,_,_=calcul_indices_passage_corr_power_spectrum_kernel(Nspectrenoyau,(Nthetas,4*p+1),thetas)
    #calcul des autocorrelations de projections du gradient suivant theta
    # sur l'axe theta
    cinit=calcul_correlations_initiales(img,thetas,p)
    # Deconvoluer légèrement les autocorrélations pour suppprimer un
    # "flou intrinsèque
    cdeconv=deconv_intrinsic_blur(cinit)
    # Calcul des supports initiaux
    supports=initial_support_estimation(cdeconv,2*p,thetas,kappa=30)
    # En déduire les autocorrélations puis le spectre de puissance de h
    hpower=Estimate_h_correlations(cdeconv,supports)
    H2=spectre_puissance_depuis_corrs(hpower,Nspectrenoyau,indexs)
    # if verbose:
    #     viewimage(fftshift(H2),titre='densite_spectrale_de_puissance')
    #     g=SinglePhaseRetrieval(H2, p,Mh=Nspectrenoyau,Ninner=Ninner)
    #     viewimage(g,titre='premier_noyau_estime')
    #     print('temps totale de la première phase', time.time()-t0)
    # boucle pour affiner le noyau
    # Seul le support induit par le noyau amélioré est utilisé.
    # On suppose que trois itération de cette boucle suffisent pour atteindre
    # l'optimum de ce que peut faire la méthode
    # on va quand meme stocker tous les trois noyaux dans une liste
    gbests=[]
    for m in range(Nouter):
        new_corrs=Estimate_h_correlations(cdeconv,supports)
        H2=spectre_puissance_depuis_corrs(new_corrs,Nspectrenoyau,indexs)
        cbest=None
        P=Propose_patch_haute_variance(imgvars,img,taille_patch)
        for k in range(Ntries):
            if verbose:
                print('boucle numéro',m, 'essai',k, 'sur ', Ntries)
            g=SinglePhaseRetrieval(abs(H2)**0.5,p,Mh=Nspectrenoyau)
            if (p//2)*2==p:

                gdeconv=np.zeros((p+1,p+1))
                gdeconv[:p,:p]=g
            else:
                gdeconv=g

            tmpim=tv_deconv(P,gdeconv, lam=1000,gamma=5,max_iters=40)[0]
            c=score_restau(tmpim)

            if verbose:
                print('score',c)
            if cbest is None or c<cbest:
                gbest=g.copy()
                cbest=c
            g=np.fliplr(np.flipud(g))
            if (p//2)*2==p:
                gdeconv=np.zeros((p+1,p+1))
                gdeconv[:p,:p]=g
            else:
                gdeconv=g

            tmpim=tv_deconv(P,gdeconv,lam=1000,gamma=5,max_iters=40)[0]
            c=score_restau(tmpim)
            if verbose:
                print('score flip',c)
            if c<cbest:
                gbest=g.copy()
                cbest=c
        supports=Restimation_supports_noyau(gbest,p,thetas,ratio=0.05)
        # if verbose:
        #     viewimage(gbest,titre='tentative'+str(m))
        #     print('temps total de la boucle numéro',m ,'est ',time.time()-t0)
        gbests.append(gbest)
    return gbest,gbests



def centrer_le_noyau(K):
    # opération optionnelle
    p=K.shape[0]
    assert p%2==1, 'Le noyau doit être de taille impaire'
    X,Y= np.meshgrid(np.arange(p),np.arange(p))
    xm=int(np.round((X*K).sum()))
    ym=int(np.round((Y*K).sum()))
    Knew=np.zeros(K.shape)
    dx=min(p-1-xm,xm)
    dy=min(p-1-ym,ym)
    #print (xm,ym,dx,dy)
    Knew[p//2-dy:p//2+dy+1,p//2-dx:p//2+dx+1]=K[ym-dy:ym+dy+1,xm-dx:xm+dx+1]
    print('pourcentage de masse perdue', (1-Knew.sum()/K.sum())*100,'%')
    Knew/=Knew.sum()
    return Knew
