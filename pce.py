import numpy as np
from scipy.special import legendre, hermitenorm
from quad4pce import PceSmolyakGrid
from scipy.interpolate import griddata
import util4pce as utl
from timeit import default_timer as timer

class PolyChaos():
    """
    POLYCHAOS is a class that makes it possible to perform uncertainty quantification
    by means of the Polynomial Chaos Expansion method.

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 17.12.2018
    HISTORY:
    """

    def __repr__(self):
        return f"PolyChaos: dim={self.dim}, order={self.order}"
    

    def __str__(self):
        # aux variables
        kmax = 5
        msg = ""
        
        msg += "Polynomial Chaos Expansion\n"
        msg += "--------------------------\n"
        msg += f"    dimensions: {self.dim}\n"
        msg += f"    order: {self.order}\n"
        
        msg += "    distrib: ["
        for k, ele in enumerate(self.distrib):
            if k > kmax - 1:
                msg += ' ... '
                break
            elif k == len(self.distrib)  - 1:
                msg += ele.upper()
            else:
                msg += ele.upper() + " , "
        msg += "]\n"

        msg += "    param: ["
        for k, ele in enumerate(self.param):
            if k > kmax - 1:
                msg += ' ... '
                break
            elif k == len(self.param)  - 1:
                msg += str(ele)
            else:
                msg += str(ele) + " , "
        msg += "]\n"

        msg += "    coeff: ["
        if self.coeff.size == 0:
            msg += " to be computed "
        else:
            for k, ele in enumerate(self.coeff):
                if k > kmax - 1:
                    msg += ' ... '
                    break
                elif k == len(self.param) - 1:
                    msg += str(ele)
                else:
                    msg += str(ele) + " , "
        msg += "]\n"
        
        return msg


    def __init__(self, order, distrib, param):

        # check input dimension
        if len(distrib) != len(param):
            raise ValueError('distrib and param must have the same length')
        
        # assign main properties
        self.dim = len(distrib)
        self.order = order
        self.distrib = distrib
        self.param = param
        self.coeff = np.empty(0)
        self.grid = None
        self.mu = None
        self.sigma = None

        (self.nt,
         self.multi_index,
         self.basis) = self.create_instance(order, distrib)
    

    def create_instance(self, order, distrib):
        """
        CREATE_INSTANCE creates and instance of the PolyChaos Class

        Paramaters
        ----------
        order (int) :
            order of the polynom
        distrib (list):
            type of distrubution ('u' for uniform and 'n' for normal)

        Returns
        -------
        nt (int) :
            size of the multivariate polynomials basis
        multi_index (ndarry):
            multi index useful (referred to the multivariate polynomials basis)
        basis (function):
            multivariate polynomials basis generator
        
        
        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 17.12.2018
        HISTORY:
        """

        # generate multi index
        multi_index, nt = self.generate_multi_index(order)

        # create multivariate polynomials basis 
        def basis(index, eps):
            """
            BASIS is the multivariate basis

            Parameters
            ----------
            index (int) :
                index of the basis (from 0 to nt)
            eps (int) :
                evaluation points:
                * from -1 to +1 for a uniform distribution,
                * from -inf to onf for a normal distribution.
            
            Return
            ------
            y (float) :
                value of the basis in eps --> y = psi(eps)
            
            AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
            DATE: 17.12.2018
            HISTORY:
            """
            # check input
            if index > (nt - 1):
                raise ValueError(f'max index possible is nt-1={self.nt-1}')
            if isinstance(eps, (list, tuple)):
                eps = np.array(eps)

            # get multi-index
            i = self.multi_index[index]
            
            # initialize output
            y = np.ones(eps.shape[0])
            # cycle on distrubutions (nt element)
            for k, dist in enumerate(distrib):
                if dist.upper() == 'U':
                    y = y * np.polyval(legendre(i[k]), eps[..., k])
                elif dist.upper() == 'N':
                    y = y * np.polyval(hermitenorm(i[k]), eps[..., k])
            
            return y
        
        return nt, multi_index, basis

        
    def generate_multi_index(self, order):
        """
        GENERATE_MULTI_INDEX provides all multi-indexes [i1, i2, ..., i_dim] whose sum <= order.
        Based on Kaarnioja2013 (program C.2) and Gerstner2007 (algorithm 8.1.1)

        Parameters
        ----------
        order (int) :
            order of the polynom
        
        Returns
        -------
        index (ndarray) :
            2D-ndarray including all multi-indexes with sum <= val
        cnt (int) :
            number of multi-indexes in index

        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 12.12.2018
        HISTORY:
        """

        index = np.empty((0, self.dim), dtype=int)
        cnt = 0

        for val in range(order + 1):
            k = np.zeros(self.dim,dtype=int)
            khat = np.ones_like(k) * val    
            p = 0

            while k[self.dim - 1] <= val:
                if k[p] > khat[p]:
                    if (p + 1) != self.dim:
                        k[p] = 0
                        p += 1
                else:
                    khat[:p] = khat[p] - k[p]
                    k[0] = khat[0]
                    p = 0
                    cnt += 1
                    index = np.append(index, k.reshape(1,-1), axis=0)
                
                k[p] = k[p] + 1
        
        return index, cnt


    def norm_factor(self, multi_index):
        """
        NORM_FACTOR returns the normalization factor for the computation
        of the PCE coefficients.

        Parameters
        ----------
        multi_index (ndarray) :
            multidimensional index related to the multivariate basis of the PCE

        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 29.12.2018
        HISTORY:
        """

        factor = 1
        for k, index in enumerate(multi_index):
            if self.distrib[k].upper() == 'U':
                factor = factor * (2 * index + 1) / 1
            elif self.distrib[k].upper() == 'N':
                factor = factor / np.math.factorial(index)
        
        return factor

    
    def spectral_projection(self, fun, level, verbose='y'):
        """
        SPECTRAL_PROJECTION computes the PCE coefficient using the
        spectral projection method.

        Parameters
        ----------
        fun (function) :
            python function to be analyzed
        level (int) :
            level of the integration (used to generate the Smolyak Grid)
        verbose (str) :
            flag to enable informative text (default is 'y', i.e. enabled)

        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 29.12.2018
        HISTORY:
        """

        # create sparse grid
        # ------------------
        if verbose == 'y':
            t1 = timer()
            print("* generation of smolyak sparse grid ... ", end=' ', flush=True)
        #
        self.grid = PceSmolyakGrid(self, level)
        #x, eps, weight, unique_x = smolyak_sparse_grid(self, level)
        #
        if verbose == 'y':
            t2 = timer()
            tel, unit = utl.human_readable_time(t2 - t1)
            print(f"done {tel :.3f} " + unit)

        # evaluate function at unique points
        # ----------------------------------
        if verbose == 'y':
            t1 = timer()
            print(f"* evaluation of the function at {self.grid.unique_x.shape[0]} unique points ... ", end=' ', flush=True)
        #
        unique_y = fun(self.grid.unique_x)
        #
        if verbose == 'y':
            t2 = timer()
            tel, unit = utl.human_readable_time(t2 - t1)
            print(f"done {tel :.3f} " + unit)
        
        # evaluate function at all points
        # -------------------------------
        if verbose == 'y':
            t1 = timer()
            print(f"* interpolation at {self.grid.x.shape[0]} points ... ", end=' ', flush=True)
            #
        y = griddata(self.grid.unique_x, unique_y, self.grid.x, method='nearest')
        #
        if verbose == 'y':
            t2 = timer()
            tel, unit = utl.human_readable_time(t2 - t1)
            print(f"done {tel :.3f} " + unit)
        
        # coefficient computation
        # -----------------------
        if verbose == 'y':
            t1 = timer()
            print("* coefficient computation ... ", end=' ', flush=True)
        #
        self.coeff = np.zeros(self.nt)
        for k in range(self.nt):
            factor = self.norm_factor(self.multi_index[k])
            self.coeff[k] = factor * np.sum(y.flatten() * self.basis(k, self.grid.eps).flatten() * self.grid.weight.flatten())
        #
        if verbose == 'y':
            t2 = timer()
            tel, unit = utl.human_readable_time(t2 - t1)
            print(f"done {tel :.3f} " + unit)
    

    def norm_fit(self, plot='n'):
        """
        NORM_FIT computes 'mean' and 'standard deviation' using the
        PCE coefficient

        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 29.12.2018
        HISTORY:
        """

        # evaluation ation of the mean
        self.mu = self.coeff[0]
        
        # evaluation ation of the standard deviation
        c_quad = self.coeff[1:] ** 2
        psi_quad = np.array([1/self.norm_factor(k) for k in self.multi_index[1:]])
        self.sigma = np.sqrt(np.sum(c_quad * psi_quad))

        if plot.lower() == 'y':
            # import plot libraries
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            
            # check if mpl is interactive
            if not mpl.is_interactive():
                plt.ion()

            # create data
            x = np.linspace(-4*self.sigma, 4*self.sigma, 501) + self.mu
            y = 1 / np.sqrt(2 * np.pi * self.sigma**2) * np.exp(-(x - self.mu)**2 / (2 * self.sigma**2))

            # plot
            plt.plot(x, y , linewidth=2)
            plt.xlabel('x', fontsize=14)
            plt.ylabel('y', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            # text box
            ax = plt.gca()
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, linewidth=2)
            ax.text(0.05, 0.95, f"$\mu={self.mu :.4f}$\n$\sigma={self.sigma  :.4f}$",
                    verticalalignment='top', bbox=props, fontsize=14, transform=ax.transAxes)
            
            # set grid and layout
            plt.grid()
            plt.tight_layout()
