import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import spsolve
from pylab import * 
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

import sys, os
from scipy.io import wavfile



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
np.seterr(over='raise')


class cburgers:

        
        
        
    def __init__(self, N = None, dt = None, alpha = 1, mu = 0, nu = 1e-5, T = 1, X = 1., F0 = 10., nComp = 1, bore = None,
                 init = None, output = None, w_len = None, w_step = None, callback = lambda *args: None):
        
        # preprocessing
        self.N = N
        self.X = X
        self.T = T
        self.mu = mu
        self.nu = nu
        self.alpha = alpha
        self.F0 = F0
        
        self.w_len = w_len
        self.w_step = w_step
        
        self.nComp = nComp
        self.isSet = False
        
        self.dt = dt
        self.implicit = False
        self.bore = bore
        
        # in_processing
        self.callback = callback
        
        # postprocessing
        self.init = init
        self.save = None
        
        if self.bore != None :
            with open(self.bore, "r") as f :
                L = [l.rstrip('\n').strip().replace('  ', ' ') for l in f.readlines()[:-1]]
                ttemp = np.cumsum([float(l.split(' ')[2]) for l in L])
                Dtemp = np.array([float(l.split(' ')[1]) for l in L])
                
                self.T = ttemp[-1]
        
                t = np.arange(0, self.T, self.dt)
                D = np.zeros_like(t)
                
                P = np.polyfit(ttemp, Dtemp, deg = 20)
                #D = np.polyval(P, t)
                #plt.plot(t, D)
                #plt.show()
                
                #self.dlogD = np.zeros_like(D)
                #for i in range(0, len(ttemp)-1, 1) :
                #    P = np.polyfit(ttemp[max(i-diffT, 0):min(i+diffT, len(ttemp))], Dtemp[max(i-diffT, 0):min(i+diffT, len(ttemp))], deg = diffT+1)
                #    I = np.where((t >= ttemp[i])*(t <= ttemp[i+1]))
                #    dP = np.polyder(P)
                #    self.dlogD[I] = np.polyval(dP, t[I])/np.polyval(P, t[I]) 
        
                dP = np.polyder(P)
                self.dlogD = np.polyval(dP, t)/np.polyval(P, t)     
            
        else :
            self.dlogD = np.zeros(int(self.T/self.dt))

    def set_matrix(self):
        print('Remplissage des matrices...', self.N)
        
        one = np.empty(self.N+1); one.fill(1)
        
        
        self.A = 2./3*sp.sparse.spdiags(one, 0, one.size, one.size)+1./6*sp.sparse.spdiags(one, 1, one.size, one.size)+1./6*sp.sparse.spdiags(one, -1, one.size, one.size)

        self.Bp = self.alpha/6*(sp.sparse.spdiags(one, 0, one.size, one.size)+sp.sparse.spdiags(one, 1, one.size, one.size)+sp.sparse.spdiags(one, -1, one.size, one.size))
        self.Bm = -sp.sparse.spdiags(one, 1, one.size, one.size)+sp.sparse.spdiags(one, -1, one.size, one.size)
    
        self.C = self.nu*(2*sp.sparse.spdiags(one, 0, one.size, one.size)- sp.sparse.spdiags(one, 1, one.size, one.size)-sp.sparse.spdiags(one, -1, one.size, one.size))/self.dx    
 
        self.E =  -self.mu*np.sqrt(self.dx)/np.sqrt(np.pi)*( 0.88366*sp.sparse.spdiags(one, 0, one.size, one.size)-0.55411*sp.sparse.spdiags(one, -1, one.size, one.size)+0.53333*sp.sparse.spdiags(one, 1, one.size, one.size))
                    
        for n in range(2, 200):
    
            self.E = self.E-self.mu*np.sqrt(self.dx)/np.sqrt(np.pi)* sp.sparse.spdiags(one, -n, one.size, one.size)*(8*(n+2)**(2.5)-32*(n+1)**(2.5)+48*n**(2.5)-32*(n-1)**(2.5)+8*(n-2)**(2.5))/15;
            
        if self.implicit :
    #        # Implicite
            self.A = self.A+self.C*self.dt/self.dx
            
            self.C =  - self.E
        else :
            # Explicite
            self.C = self.C - self.E
        
    def reset(self) :
        self.isSet = False
        
    def mcInitial(self):
                                
        if (self.N != None) :
            self.dx = 1./self.N
            self.x = np.linspace(0., self.X, self.N+1)
            
        
        if (self.dt == None) : 
            self.dt = (self.dx**2/(12*self.nu+4*self.dx))/10
            print('Pas de temps : ', self.dt)
            
        if self._type == 'mesure' : 
                
            f=open(self.customInit[0].replace('*', '1'), "r")
                
            L = f.readlines()
            L = np.array([[float(a) for a in l.rstrip("\n").split('\t')] for l in L]) #/1.65e-5
                                    
            self.x = L[:, 0]

            self.N = len(self.x)-1
            
            self.dx = self.x[1]-self.x[0]    
                  
        self.set_matrix()    
            
        self.fps = max(1, int(self.T/(200*self.dt)))
        
        print('================ Résumé ==============================')
        print("alpha = ", self.alpha, "nu = ", self.nu, "mu = ", self.mu)
        print("Calcul implicite : ", self.implicit)
        print("T = ", self.T, "dx = ", self.dx, "dt = ", self.dt)
        print("len(X) ", len(self.x))
        print('======================================================')

                                    
    def initial(self, s=''):
        """ Initialisation  du vecteur
        s est un paramètre utilisé seulement dans le cas Montecarlo, ou il est utilisé conjointement avec customInit """
        
        if self.nComp == 1 :
                                            
            if (self.N != None) :
                self.dx = 1./self.N
                self.x = np.linspace(0., self.X, self.N+1)
                
        else :
            
            Lambda0 = self.init(s)
                
            self.N = len(Lambda0)-1
            self.dx = self.x[1]-self.x[0]
                                            
        if type(self.init) == str :
            
            
            if self.customInit.split('.')[-1] == 'wav' :
                rate, Lambda0 = wavfile.read(self.customInit)
                Lambda0 = Lambda0-Lambda0.mean()
                                
            elif self.customInit.split('.')[-1] == 'txt' :
                
                f=open(self.customInit, "r")
                Lambda0 = f.readlines()
                Lambda0 = np.array([float(i) for i in Lambda0]) #/1.65e-5
                Lambda0 -= Lambda0.mean()
        
            
        elif hasattr(self.init, '__call__') :
            Lambda0 = self.init()
            
        elif type(self.init) == np.ndarray :
            Lambda0 == self.init
            
        self.N = len(Lambda0)-1
        self.dx = self.X/self.N
        self.x = np.linspace(0., self.X, self.N+1)
            
        # Si calcul par morceaux (parallel)                
        if self.w_len != None :
            self.N = self.w_len+2*self.w_step-1
            
        if self.nComp <= 1 :
            if (self.dt == None) : 
                self.dt = (self.dx**2/(12*self.nu+4*self.dx))/10
                print('Pas de temps : ', self.dt)
            
            self.set_matrix()    
                
            self.fps = max(1, int(self.T/(200*self.dt)))
            
    #        self.mu *= np.sqrt(max(abs(Lambda0)))
    #        print(self.fps)
            
            
            print('================ Résumé ==============================')
            print("alpha = ", self.alpha, "nu = ", self.nu, "mu = ", self.mu)
            print("Calcul implicite = ", self.implicit)
            print("dt = ", self.dt, ",N = ", self.N, ',T = ', self.T)
            print('======================================================')
        
        return Lambda0
        
        
        
    def inside_opt(self, density = False, spectrum = False):
        
        self.in_density = density
        self.in_spectrum = spectrum
        
        
    def out_opt(self, density = False, movie = False, save = None):
        
        self.density = density
        self.movie = movie
        self.save = save
        
    def _prep_movie(self) :
        FFMpegWriter = animation.writers['avconv']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        self.writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=1800)
        fig = plt.figure(1, figsize=(14, 6))
        
        self.writer.setup(fig=fig, outfile = 'writer_test.mp4', dpi=100)
        
#        self.writer.saving(self.fig, "writer_test.mp4", 100)
        
        
        
        
    def _plot_movie(self, writer, Lambda, LambdaM) :
        
                
        plt.clf()
        
        if hasattr(self.movie, '__call__') :
           self.movie(self.x, Lambda)        

        plt.draw()
        self.writer.grab_frame()
        
    def _finish(self, Lambda) :
                
        if self.movie != None :
            print("sauvegarde de l'animation")
            self.writer.finish()
        
        if self.nComp < 2 and self.save != None :
            print('Sauvegarde du fichier ', self.save)
            
            if self.save.split('.')[-1] == "npy" :
                np.save(self.save, Lambda)
            if self.save.split('.')[-1] == "csv" :
                np.savetxt(self.save, Lambda, delimiter=',')
                
            if self.save.split('.')[-1] == "txt" :
                np.savetxt(self.save, Lambda, delimiter=' ')
                
            if self.save.split('.')[-1] == "wav" :                
                wavfile.write(self.save, 44100, Lambda.astype(np.float32)/max(abs(Lambda)))  
          
        
    def FEM(self, Lambda = []):
        
        if self.movie :
            self._prep_movie()
            
        LambdaM = 2*np.max(np.abs(Lambda))
   
        
        t = 0
        n = 0

        while (t < self.T):
            
            try :
            
                if self.nComp == 1 :
                    sys.stdout.write('t = %.11f \r' % t)
                    sys.stdout.flush()
            
                #try:
            
                # explicite
                Lambda = (1+self.dx*self.dlogD[n])*Lambda+self.dt*spsolve(self.A, -self.C*Lambda+(self.Bm * Lambda)*(self.Bp*Lambda))/self.dx

                # implicite
#                    Lambda = Lambda+self.dt*spsolve(self.Ai, (self.Bm * Lambda)*(self.Bp*Lambda))/self.dx

                #funci = lambda y: A*(-y+Lambda)+dt*(-nu*C*(y+Lambda)/2+(Bm * (y+Lambda))*(Bp*(y+Lambda))/(4.*dx))
                #Lambda = fsolve(funci, Lambda)
            
                Lambda[0]=0
                Lambda[len(Lambda)-1]=0
                t += self.dt
                n += 1
            
            except Exception as err:
            
                print(err, 'en t = ', t)
                
                self._finish(Lambda)
                
                raise Exception('Problème de stabilité')
                return Lambda
               
                    
            except KeyboardInterrupt:
                print('Interruption control C en t = ', t)
                self._finish(Lambda)
                return Lambda
            
            if self.movie and (n%self.fps == 0):
                self._plot_movie(self.writer, Lambda, LambdaM)
                
        if self.nComp == 1 :
                
            self._finish(Lambda)
    
        return(Lambda)
        

    def partialmcFEM(self, s) :
        
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        
        Lambda0 = self.initial(s)
                     
        Lambda = self.FEM(Lambda0)
                
        if self.save != None :

             np.savetxt(self.customInit[0].replace('*', str(s)+self.save), Lambda, delimiter=',')
                          
        return(Lambda)

    # Monte Carlo FEM    
    def mcFEM(self):
        
#        sys.stdout = open(os.devnull, 'w')
        
        pool = Pool(processes=min(cpu_count()-2, self.nComp)) 
        partialFEM = partial(self.partialmcFEM)
        S = pool.map(partialFEM, range(1, self.nComp+1))
        pool.close()
        pool.join()
        
        sys.stdout = sys.__stdout__
        
        S = np.vstack(S).transpose()
        
        self._finish(S)
        
        return S
    
    """ PArallel computation """
    
    def partialParFEM(self, s) :
        
        if s == 0 :
            
            Lambda = self.FEM(self.Lambda0[:self.w_len+2*self.w_step])
        else :
            Lambda = self.FEM(self.Lambda0[s*self.w_len-self.w_step:(s+1)*self.w_len+self.w_step])
                          
        return(Lambda)
    
    # Parallel computation
    def parFEM(self, Lambda0):
        
        self.Lambda0 = Lambda0
                      
        nbIter = len(Lambda0)//(self.w_len)
        
        pool = Pool(processes=min(cpu_count()-2, nbIter))
        partialFEM = partial(self.partialParFEM)
        S = pool.map(partialFEM, range(0, nbIter))
        pool.close()
        pool.join()
    
        S = np.concatenate((S[0][:self.w_len], np.concatenate([s[self.w_step:-self.w_step] for s in S[1:]])))
        
        self._finish(S)
        
        return S

        
import time
if __name__ == "__main__":
    
    print('temps de calcul :')
  
