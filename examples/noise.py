import numpy as np
import scipy as sp
from scipy.special import erf   
from numpy.fft import fft


""" Bruit rouge (pont brownien) """
def redNoise(x):
    
    #x = 1.0*numpy.arange(N+1)/N
    B = np.cumsum(np.append([0], np.random.uniform(-1, 1, x.size-1)))*np.sqrt(x[-1]/len(x))

    B = B-x/x[-1]*B[-1]
    return(B)
    
""" Bruit blanc filtre """
def whiteNoise(x, f0, f1):
    
    B = np.random.normal(-1, 1, x.size)
    
    Bp = np.fft.fft(B)
    
    f0 = int(f0*x.max())
    f1 = int(f1*x.max())
    # Filtre coupe bas    
    Bp[0:f0] = 0
    Bp[-f0-1:] = 0
    
    # filtre coupe haut
    Bp[f1:-f1] = 0
    
    B = np.fft.ifft(Bp)
    
    return(B.real)
    
""" Calcul de la covariance """

def proc(S, trunc = 1):
        
    mean = np.mean(S, 0)
    
    std = np.std(S, 0)
    
    print(S.shape[0]//trunc)
    
    cov = np.zeros((S.shape[0]//trunc-1, S.shape[0]//trunc-1))
    
    for x in range(0, S.shape[0]//trunc-1):
        for y in range(0, S.shape[0]//trunc-1):
            
            cov[x, y] = np.mean(S[trunc*x,:]*S[trunc*y, :])-np.mean(S[trunc*x, :])*np.mean(S[trunc*y, :])
        
    return cov
    
import matplotlib.pyplot as plt

def narrowBandNoise(t, tc=1e3, F0 = 10, sigma = 1):
    
    print('Création du bruit bande étroite (Matrice de covariance)...')
    N = t.size
    omega0 = 2*np.pi*F0

    matcov = np.zeros((N,N))
    
    dt = t[1]-t[0]
    
    for i in range(N) :
        j = np.arange(i,N, 1)
        matcov[i, j] = sigma**2*np.exp(-dt*abs(i-j)/tc)*np.cos(omega0*dt*abs(i-j))
        matcov[j, i] = matcov[i, j]
        
    L = np.linalg.cholesky(matcov)
    
    noise = np.random.normal(0, 1, N)
    
    noise = L.dot(noise)
    
    # Continuité vers 0 aux bords
    #noise[0:100] *= np.arange(0, 1, 1/100)
    
    #noise[-100:] *= np.arange(1, 0, -1/100)
    
    plt.plot(noise)
    plt.show()
        
    return(noise)
    
    
def WoodChan(t, tc=1e6, F0 = 10, sigma = 1.):
    
    print('Création du bruit bande étroite (Wood et Chan)...')
    N = t.size
    omega0 = 2*np.pi*F0
    
    #print('dt = ', t[1]-t[0], 1./N)
    dt = 1/N #t[1]-t[0]

    k = np.arange(0,N+1, 1)
    autocov = sigma**2*np.exp(-dt*abs(k)/tc)*np.cos(omega0*dt*abs(k))
    
    #print(N, tc, F0, omega0, sigma, dt, t)

    
    # Vérifier les indices ???)
    ligne1C = autocov[np.concatenate((np.arange(0, N+1, 1), np.arange(N-1, 0, -1)))]
    lambdak = np.real(fft(ligne1C)).astype(complex)
    #lambdak = fft(ligne1C)
    
    #print(lambdak)
    
    zr = np.random.normal(0, 1, N+1)
    zi = np.random.normal(0, 1, N-1)
    
    zr[0] *= np.sqrt(2)
    zr[N] *= np.sqrt(2)
        
    zr = np.concatenate((zr[0:N+1], zr[range(N-1, 0, -1)]))
    zi = np.concatenate(([0], zi, [0] , -zi[range(N-2, -1, -1)]))
            
    z = np.real(np.fft.fft((zr+1j*zi)*np.sqrt(lambdak)))
    
    noise = z[0:N]/(2*np.sqrt(N))
    
    # Lissage sur les bords
    
    # Continuité vers 0 aux bords
    noise[0:int(F0)] *= (1+erf(np.arange(-2, 2, 4/int(F0))))/2
    
    noise[-int(F0):] *= (1+erf(np.arange(2, -2, -4/int(F0))))/2

    
#    plt.subplot(121)
#    plt.plot(t, noise)
#    plt.subplot(122)
#    plt.plot(np.abs(np.fft.fft(noise)))    
#    plt.show()
    return(noise)
        
    
    
#import rfp
    
if __name__ == "__main__":
    
    ech = 10000
    F0 = 100.
    #noise  = narrowBandNoise(np.arange(0, 2, 1/ech), F0=F0)
    noise = WoodChan(np.arange(0, 2, 1/ech), F0=F0)
    noise /=  max(np.abs(noise))
    #plt.plot(noise)
    corr = np.correlate(noise, noise, mode='full')
    #plt.plot(corr)
    
    spctr = np.fft.fft(noise)
    
    F= np.arange(0, ech, 1)/2
    
    indexF0 = int(F0*2)

    deltaF = indexF0//2
    
    for n in range(1, 3) :
        
        #try :
    
            C, D, e = rfp.modan(1, 1, F[n*indexF0-deltaF:n*indexF0+deltaF], spctr[n*indexF0-deltaF:n*indexF0+deltaF])

            #print(rfp.computePoles(C, D))
    
            plt.semilogy(F[n*indexF0-deltaF:n*indexF0+deltaF], np.abs(spctr[n*indexF0-deltaF:n*indexF0+deltaF]))

            plt.semilogy(F[n*indexF0-deltaF:n*indexF0+deltaF], np.abs(C.eval(F[n*indexF0-deltaF:n*indexF0+deltaF])/D.eval(F[n*indexF0-deltaF:n*indexF0+deltaF])))
            
            #erreur = np.linalg.norm(C.eval(F[n*indexF0-deltaF:n*indexF0+deltaF])/D.eval(F[n*indexF0-deltaF:n*indexF0+deltaF])-spctr[n*indexF0-deltaF:n*indexF0+deltaF])/np.linalg.norm(spctr[n*indexF0-deltaF:n*indexF0+deltaF])
            
            print("erreur", e)
            
        #except :
        #    pass
    
    plt.show()
        
    
    #print(rfp.computePoles(C, D))
    
    
    # Sauvegarde du fichier bruit
    #with open('input.txt', 'w') as myFile :
    #    for i in range(len(noise)) :
    #        myFile.write(str(noise[i])+'\n')

    
    
