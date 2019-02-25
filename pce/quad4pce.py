import numpy as np
from pce.util4pce import columnize
from scipy.special import binom


def clencurt(rule=5, method='FFT'):
    """
    CLENCURT provides points and weight according to the Clenshaw-Curtis methods.
    
    Parameters
    ----------
    rule (int) :
        number of points (defautl is 5)
    method (str) :
        FFT or explicit (default is FFT)
    
    Returns
    -------
    x (ndarray) :
        1D-ndarray with points (between -1 and 1)
    w (ndarray) :
        1D-ndarray with weights (sum of all weights is 2)
    
    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 03.12.2018
    HISTORY:
    """
    
    if method.upper() == 'FFT': # FFT (Greg von Winckel, Mathworks https://goo.gl/6JVJXN)
        
        if rule == 1:
            x = np.array([0])
            w = np.array([2])
        else:
            n = rule - 1
            c = np.zeros((rule, 2))
            c[:rule:2,0] = 2 / np.append([1], np.array([1-np.arange(2,rule,2)**2]))
            c[1,1] = 1
            f = np.fft.ifft(np.concatenate((c[:rule,:], c[-2:0:-1,:]),axis=0), axis=0).real
            w = 2 * np.concatenate(([f[0,0]], 2*f[1:rule-1,0], [f[rule-1,0]]))/2
            x = (rule-1) * f[:rule,1]
            x = x[-1::-1]
        return x, w
    
    elif method.upper() == 'EXPLICIT':   # explicit way (Waldvogel2006, eq. 2.1 and 2.4)
        if rule == 1:
            x = np.array([0])
            w = np.array([2])
        else:
            n = rule - 1
            # build points
            theta = np.arange(rule) * np.pi / n
            x = np.cos(theta)
            w = 0

            # build weights (see Waldvogel2003)
            c = np.ones((rule))
            c[1:-1] = 2
            j = np.arange((n/2)//1) + 1
            
            b = 2 * np.ones(j.shape)
            if np.mod(n/2,1) == 0:
                b[int(n/2) - 1] = 1
            
            j.shape = (1, -1)
            b.shape = (1, -1)
            j_theta = j * theta.reshape(-1,1)
            w = c/n * (1 - np.sum(b/(4*j**2 - 1) * np.cos(2*j_theta), axis=1))

            # reorder points in ascending order (not necessary for weights)
            x = np.round(x[-1::-1] * 1e13) / 1e13


def quad_coeff(rule=5, kind='GL'):
    """
    QUAD_CEOFF provides points and weights for a numerical integration.
    Gauss-Legendre and Hermite quadrature are supported up to rule = 10
    Clenshaw-Curtis is supported up tu rule = 17

    Parameters
    ----------
    rule (int) :
        number of integration points (default is 5)
    kind (int) :
        kind of integration
        GL for Gauss-Legendre (default)
        GH for Gauss-Hermite
        CC for Clenshaw-Curtis
    
    Returns
    -------
    x (ndarray) :
        1D-ndarray with points
    w (ndarray) :
        1D-ndarray with weights
    
    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 15.11.2018
    HISTORY:
    07.12.2018 based on numpy function for Legendre and Hermite
    07.12.2018 replaced lookup table with function for Clenshaw-Curtis
    24.12.2018 changed string to define king of integration
    """

    if kind.upper() == 'GL':
        x, w = np.polynomial.legendre.leggauss(rule)
        
    elif kind.upper() == 'GH':
        x, w = np.polynomial.hermite.hermgauss(rule)
        
    elif kind.upper() == 'CC':
        x, w = clencurt(rule)

    return x, w


class PceSmolyakGrid():
    """
    PCESMOLYAKGRID is a class to handle sparse Smolyak Grid in the framework
    of the Polynomial Chaos Expansion.

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 27.12.2018
    HISTORY:
    """

    def __init__(self, polynom, level):
        
        (self.x,
         self.eps,
         self.weight,
         self.unique_x,
         self.inverse_index) = self.smolyak_sparse_grid(polynom, level)


    def smolyak_sparse_grid(self, polynom, level):
        """
        SMOLYAK_SPARSE_GRID builds a spase grid for multi-dimensional integration.
        It is based on Clenshaw-Curtis points and weights with growth rule: 2**(l-1) + 1.

        Parameters
        ----------
        polynom (PolyChaos) :
            PolyChaos instance
        level (int) :
            level of the integration

        Results
        -------
        x_points (ndarray) :
            2D-ndarray with all integration points (with repetition)
        eps_points (ndarray) :
            2D-ndarray with all integration points (with repetition) in the suitable range
            for the othogonal polynomials
        weights (ndarray) :
            2D-ndarray with weights to be used with points (with repetition)
        unique_points (ndarray) :
            2D-ndarray with non-repeated integration points


        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 24.12.2018
        HISTORY:
        """

        # min and max level
        o_min = max([level + 1, polynom.dim])
        o_max = level + polynom.dim
        
        # get multi-index for all level
        comb = np.empty((0, polynom.dim), dtype=int)
        for k in range(o_min, o_max+1):
            multi_index, _ = self.index_with_sum(polynom.dim, k)
            comb = np.append(comb, multi_index, axis=0)
        
        # initialize final array
        x_points = np.empty((0, polynom.dim))
        eps_points = np.empty((0, polynom.dim))
        weights = np.empty((0,))

        # define integration points and weights
        for lev in comb:
            local_x = []
            local_eps = []
            local_w = []
            coeff = (-1)**(level + polynom.dim - np.sum(lev)) * binom(polynom.dim - 1, level + polynom.dim - np.sum(lev))
            
            # cycle on integration variables
            for l, k, p in zip(lev, polynom.distrib, polynom.param):
                # set integration type depending on distrib
                if k.upper() == 'U':
                    kind = 'CC'
                elif k.upper() == 'N':
                    kind = 'GH'
                
                # get number of integration points
                n = self.growth_rule(l, kind=kind)
                # get gauss points and weight
                x0, w0 = quad_coeff(rule=n, kind=kind)

                # change of variable
                if (k.upper() == 'U'):
                    eps = np.copy(x0)
                    w = np.copy(w0) * 0.5
                    if p != [-1, 1]:
                        x = (p[1] - p[0]) / 2 * x0 + (p[1] + p[0]) / 2
                    else:
                        x = np.copy(x0)
                    
                elif (k.upper() == 'N'):
                    eps = np.sqrt(2) * 1 * x0 + 0 
                    x = eps * p[1] + p[0]
                    w = w0 * (1 / np.sqrt(np.pi))

                # store local points
                local_x.append(x)
                local_eps.append(eps)
                local_w.append(w)
            
            # update final array
            x_points = np.concatenate((x_points, np.concatenate(columnize(*np.meshgrid(*local_x)), axis=1)))
            eps_points = np.concatenate((eps_points, np.concatenate(columnize(*np.meshgrid(*local_eps)), axis=1)))
            weights = np.concatenate((weights, coeff * np.prod(np.concatenate(columnize(*np.meshgrid(*local_w)), axis=1),axis=1)))
        
        # get unique points
        unique_x_points, inverse_index = np.unique(x_points, axis=0, return_inverse=True)

        return x_points, eps_points, weights, unique_x_points, inverse_index


    def index_with_sum(self, dim, val):
        """
        INDEX_WITH_SUM provide all multi-indexes [i1, i2, ..., i_dim] whose:
            * sum is val
            * sum >= dim
        Based on Kaarnioja2013 (program C.2) and Gerstner2007 (algorithm 8.1.1)
        The algorithm is modified to provide also [1, 1, ..., 1] if val==dim.

        Parameters
        ----------
        dim (int) :
            dimension of the multi-index
        val (int) :
            target sum of the multi-index
        
        Returns
        -------
        index (ndarray) :
            2D-ndarray including all multi-indexes with sum = val
        cnt (int) :
            number of multi-indexes in index

        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 03.12.2018
        HISTORY:
        """

        # check feasibility
        if val < dim:
            raise ValueError(f'dim={dim} --> minimum value for val=dim={dim} (here {val} is provided)')

        k = np.ones(dim,dtype=int)
        khat = k * (val - dim + 1)

        index = np.empty((0, dim), dtype=int)

        cnt = 0
        p = 0

        while k[dim - 1] < val:
            if k[p] > khat[p]:
                if (p + 1) != dim:
                    k[p] = 1
                    p += 1
            else:
                khat[:p] = khat[p] - k[p] + 1
                k[0] = khat[0]
                p = 0
                cnt += 1
                index = np.append(index, k.reshape(1,-1), axis=0)
            
            k[p] = k[p] + 1
        return index, cnt


    def growth_rule(self, level, kind='Legendre'):
        """
        GROWTH_RULE generates e sequence of integration point and weights for a given level.
        Legendre and Hermite follow a linear increase.
        Clenshaw-Curtis follows a non-linear increase.
        (ref. Eldred2009)

        Parameters
        ----------
        level (int) :
            desired level (minimum value is 1)
        kind (str) :
            'GL' or 'GH' or 'CC' (case insensitive)
        
        Returns
        -------
        n (int) :
            number of integration points (i.e. degree of the integration)

        AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
        DATE: 07.12.2018
        HISTORY:
        24.12.2018 changed string to define king of integration
        """
        method_map = {'GL':1, 'GH':1, 'CC':2}

        if method_map[kind.upper()] == 1: 
            n = 2*level - 1
        elif method_map[kind.upper()] == 2:
            n = 1 if level == 1 else 2**(level - 1) + 1
        
        return n