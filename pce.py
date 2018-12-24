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


    def __init__(self, dim, order, distrib, param):
        
        self.dim = dim
        self.order = order
        self.distrib = distrib
        self.param = param
        self.coeff = np.empty(0)

        (self.nt,
         self.multi_index,
         self.basis) = self.create_instance(dim, order, distrib)
    

    def create_instance(self, dim, order, distrib):
        """
        CREATE_INSTANCE creates and instance of the PolyChaos Class

        Paramaters
        ----------
        dim (int) :
            dimension of the multi-index
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
        multi_index, nt = self.generate_multi_index(dim, order)

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

        
    def generate_multi_index(self, dim, order):
        """
        GENERATE_MULTI_INDEX provides all multi-indexes [i1, i2, ..., i_dim] whose sum <= order.
        Based on Kaarnioja2013 (program C.2) and Gerstner2007 (algorithm 8.1.1)

        Parameters
        ----------
        dim (int) :
            dimension of the multi-index
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

        index = np.empty((0, dim), dtype=int)
        cnt = 0

        for val in range(order + 1):
            k = np.zeros(dim,dtype=int)
            khat = np.ones_like(k) * val    
            p = 0

            while k[dim - 1] <= val:
                if k[p] > khat[p]:
                    if (p + 1) != dim:
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


if __name__ == "__main__":
    from util import columnize
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    # parameters
    dim = 2
    order = 2
    distrib = ['u', 'n']
    param = [[-4,3],[1.2, 0.1]]

    # generate PCE
    poly = PolyChaos(dim, order, distrib, param)

    # test basis 
    npt = 100
    x1 = np.linspace(-1,1,npt)
    x2 = np.linspace(-3,3,npt)
    X = np.concatenate(columnize(*np.meshgrid(x1,x2)), axis=1)
    X1 = X[:,0].reshape(npt,npt)
    X2 = X[:,1].reshape(npt,npt)
    
    # define grid
    n1 = np.floor(np.sqrt(poly.nt))
    n2 = np.ceil(poly.nt/n1)

    h = plt.figure(figsize=(11,7))
    for k in range(poly.nt):
        ax = h.add_subplot(n1,n2,k+1, projection='3d')
        y = poly.basis(k, X)
        Y = y.reshape(npt,npt)
        ax = h.gca(projection='3d')
        ax.plot_surface(X1,X2,Y, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_title(f'$\psi_{{ {k} }}$')
        ax.text(-0.8, -7.5, ax.get_zlim()[0], 'x1', (1,1,0), fontsize=6)
        #ax.set_xlabel('x1', fontsize=6)
        ax.xaxis.set_label_coords(0, 0)
        ax.set_ylabel('x2', fontsize=6)
        ax.tick_params(labelsize=5)
        plt.tight_layout()
    
    h.savefig(f'pce_basis_dim_{dim}_order_{order}',dpi=150)
