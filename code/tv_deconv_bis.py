import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2
import scipy
from skimage import io
import time
import matplotlib.pyplot as plt
from pathlib import Path


fft = scipy.fft.rfft2
ifft = scipy.fft.irfft2
norm = np.linalg.norm

real = np.real
sin = np.sin
conj = np.conj
pi = np.pi


def pad_image(im,pad=10):
    out=np.zeros((im.shape[0]+2*pad,im.shape[1]+2*pad))
    out[pad:-pad,pad:-pad]=im
    for k in range(pad):
        out[k,pad:-pad]=im[0,:]
        out[-k-1,pad:-pad]=im[-1,:]
        out[pad:-pad,k]=im[:,0]
        out[pad:-pad,-k-1]=im[:,-1]
    out[:pad,:pad]=im[0,0]
    out[-pad:,:pad]=im[-1,0]
    out[:pad,-pad:]=im[0,-1]
    out[-pad:,-pad:]=im[-1,-1]
    return out

def unpad_image(im,pad=10):
    return im[pad:-pad,pad:-pad].copy()

def Fourier_kernel(K,s):
    assert K.shape[0]%2==1 and K.shape[1]%2==1, "Taille de noyau non impaire"
    Kf=np.zeros(s)
    Ky,Kx=K.shape
    Kx2=Kx//2
    Ky2=Ky//2
    Kf[:Ky2+1,:Kx2+1]=K[Ky2:,Kx2:]
    Kf[:Ky2+1,-Kx2:]=K[Ky2:,:Kx2]
    Kf[-Ky2:,:Kx2+1]=K[:Ky2,Kx2:]
    Kf[-Ky2:,-Kx2:]=K[:Ky2,:Kx2]
    return fft2(Kf)

def taper_image(I,K):
    """ Floute une image I par le noyau K (circulairement) cela donne une image J
    On mélange l'image I avec l'image J de manière à ce que J soit prépondérente aux bords.
    L'image J, lorsqu'on la déconle par le noyau K n'aura pas d'effets de bord. """
    kh,kw=K.shape
    Ih,Iw=I.shape
    wx=np.ones((Ih,Iw),dtype=np.float32)
    wy=np.ones((Ih,Iw),dtype=np.float32)
    X,Y=np.meshgrid(np.arange(0,Iw),np.arange(0,Ih))
    wy[:kh,:]=sin(Y[:kh,:]*pi/(2*kh-1))**2
    wy[-kh:,:]=sin((Ih-Y[-kh:,:])*pi/(2*kh-1))**2
    wx[:,:kw]=sin(X[:,:kw]*pi/(2*kh-1))**2
    wx[:,-kw:]=sin((Iw-X[:,-kw:])*pi/(2*kh-1))**2
    fK=Fourier_kernel(K,I.shape)
    J=real(ifft2(fft2(I)*fK))
    out=J*(1-wx*wy)+I*(wx*wy)
    return out

def conv(im,K,Fourierform=False):
    if not Fourierform:#on recoit les formes spatiales
        Kf=Fourier_kernel(K,im.shape)
        imf=fft2(im)
        return np.real(ifft2(imf*Kf))
    else:# forme Fourier
        return np.real(ifft2(im*K))

def champ_grad(u):#gradient circulaire
    return np.stack((np.c_[(u[:,0]-u[:,-1]).reshape(-1,1),u[:,1:]-u[:,:-1]],\
             np.r_[(u[0,:]-u[-1,:]).reshape(1,-1),u[1:,:]-u[:-1,:]]))

def universal_dot(X,Y):
    return (X*Y).sum()

def div_champ(c):
    return np.c_[c[0,:,1:]-c[0,:,:-1],(c[0,:,0]-c[0,:,-1]).reshape(-1,1)]+\
            np.r_[c[1,1:,:]-c[1,:-1,:],(c[1,0,:]-c[1,-1,:]).reshape(1,-1)]




def d_sub_problem(u,b,gamma=5/255):
    gradu=champ_grad(u)
    champ=gradu+b
    s=champ.shape[1:]
    no=(champ**2).sum(axis=0)**0.5
    mask=(no<(1/gamma))
    no[mask]=0.001
    no=no.reshape(1,*s)
    mu=1-1/(gamma*no)
    champ*=mu
    champ[:,mask]=0
    #champ[1,mask]=0
    return champ

def u_sub_problem(f,d,b,K,lamb,gamma=5,Fourierform=False,fdenom=None):
    """ si Fourierform=True alors f et K sont donnees sous forme Fourier"""
    if not Fourierform:
        ff=fft2(f)
        Kf=Fourier_kernel(K,f.shape)
    else:
        ff=f
        Kf=K
    if fdenom is None:
        Kl=np.zeros(f.shape)
        Kl[0,0]=4
        Kl[0,1]=-1
        Kl[1,0]=-1
        Kl[-1,0]=-1
        Kl[0,-1]=-1
        fdenom=real(fft2(Kl))
        fdenom+=(lamb/gamma)*(abs(Kf)**2)

    numer=conj(Kf)*ff*(lamb/gamma)-fft2(div_champ(d-b))
    return real(ifft2((numer)/fdenom))

def sym_image(x):
    out=np.concatenate((x,np.fliplr(x)),axis=1)
    out=np.concatenate((out,np.flipud(out)),axis=0) #symetrise l'image
    return out

def TV(im):
    g=champ_grad(im)
    n=((g**2).sum(axis=0))**0.5
    return n.sum()

def fonctionnelle(f,u,K,d,b,lamb,gamma=5):
    v1=TV(u)+lamb/2*norm(f-conv(f,K))**2
    v2=v1+gamma/2*norm(d-champ_grad(u)-b)**2
    return (v1,v2)

def norm(X):
    return ((X**2).sum())**0.5

def TVdeconv(im,K,lamb,nbit=140,gamma=5/255,edgehandle='taper'):
    """
    Si edgehandle= 'taper' alors on ajoute à l'image une bordure lisse
    Si edgehandle= 'sym' alors on symmetrise l'image
    Si edgehandle= 'nothing' alors on ne fait rien (mauvais)
    """
    if edgehandle=='taper':
        f=taper_image(pad_image(im,K.shape[0]),K)
    elif edgehandle=='sym':
        f=sym_image(im)
    else:
        f=im.copy()
    s=f.shape
    Kf=Fourier_kernel(K,s)
    Kl=np.zeros(f.shape)
    Kl[0,0]=4
    Kl[0,1]=-1 
    Kl[1,0]=-1
    Kl[-1,0]=-1
    Kl[0,-1]=-1
    fdenom=real(fft2(Kl))
    fdenom+=lamb/gamma*(abs(Kf)**2)
    u=np.zeros(s)
    unew=np.zeros(s)
    d=np.zeros((2,*s))
    b=np.zeros((2,*s))
    tol=norm(f)/1000
    counter=0
    ff=fft2(f)
    #Kfff=conj(Kf)*ff*(lamb/gamma)
    #print("iteration",counter,' Fonctionnelles=',\
     #         fonctionnelle(f,unew,K,d,b,lamb))

    while counter==0 or (norm(unew-u)>tol and counter<nbit):
        counter+=1
        u=unew
        d=d_sub_problem(u, b,gamma=gamma)

        unew=u_sub_problem(ff, d, b, Kf, lamb,gamma=gamma,\
                           Fourierform=True,fdenom=fdenom)
        b+=(champ_grad(unew)-d)
        #print("iteration",counter,)#' Fonctionnelles=',\
         #     fonctionnelle(f,unew,K,d,b,lamb))

    if edgehandle=='taper':
        out=unpad_image(unew,K.shape[0])
    elif edgehandle=='sym':
        out=unew[:im.shape[0],:im.shape[1]]
    else:
        out=unew
    print(counter)
    return out, counter