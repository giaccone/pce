import numpy as np
from scipy.special import legendre, hermitenorm


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
