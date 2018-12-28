import numpy as np
from scipy.special import legendre, hermitenorm
from quad4pce import PceSmolyakGrid
from scipy.interpolate import griddata
import util as utl
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
        to do ...
        """

        factor = 1
        for k, index in enumerate(multi_index):
            if self.distrib[k].upper() == 'U':
                factor = factor * (2 * index + 1) / 2
            elif self.distrib[k].upper() == 'N':
                factor = factor / np.math.factorial(index)
        
        return factor

    
    def spectral_projection(self, fun, level):
        """
        to do ...
        """

        # create sparse grid
        # ------------------
        t1 = timer()
        print("* generation of smolyak sparse grid ... ", end=' ', flush=True)
        #
        self.grid = PceSmolyakGrid(self, level)
        #x, eps, weight, unique_x = smolyak_sparse_grid(self, level)
        #
        t2 = timer()
        tel, unit = utl.human_readable_time(t2 - t1)
        print(f"done {tel :.3f} " + unit)

        # evaluate function at unique points
        # ----------------------------------
        t1 = timer()
        print(f"* evaluation of the function at {self.grid.unique_x.shape[0]} unique points ... ", end=' ', flush=True)
        #
        unique_y = fun(self.grid.unique_x)
        #
        t2 = timer()
        tel, unit = utl.human_readable_time(t2 - t1)
        print(f"done {tel :.3f} " + unit)
        
        # evaluate function at all points
        # -------------------------------
        t1 = timer()
        print(f"* interpolation at {self.grid.x.shape[0]} points ... ", end=' ', flush=True)
        #
        y = griddata(self.grid.unique_x, unique_y, self.grid.x, method='nearest')
        #
        t2 = timer()
        tel, unit = utl.human_readable_time(t2 - t1)
        print(f"done {tel :.3f} " + unit)
        
        # coefficient computation
        # -----------------------
        t1 = timer()
        print("* coefficient computation ... ", end=' ', flush=True)
        #
        self.coeff = np.zeros(self.nt)
        for k in range(self.nt):
            factor = self.norm_factor(self.multi_index[k])
            self.coeff[k] = factor * np.sum(y * self.basis(k, self.grid.eps) * self.grid.weight)
        #
        t2 = timer()
        tel, unit = utl.human_readable_time(t2 - t1)
        print(f"done {tel :.3f} " + unit)
