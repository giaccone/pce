import numpy as np

class PolyChaos():
    """
    to be done
    """

    def __init__(self, dim, order, distrib):
        pass
    
    def create_multi_index(self, order, dim):
        pass

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