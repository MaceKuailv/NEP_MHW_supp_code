import numpy as np
import xarray as xr

def llc_compact_to_faces(data_compact, less_output = False):
    """
    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it into the 5 'faces'
    of the llc grid.

    The five faces are 4 approximately lat-lon oriented and one Arctic 'cap'

    Parameters
    ----------
    data_compact : ndarray
        An 2D array of dimension  nl x nk x 13*llc x llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output


    Returns
    -------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_compact

    """

    dims = data_compact.shape
    num_dims = len(dims)

    # final dimension is always of length llc
    llc = dims[-1]

    # dtype of compact array
    arr_dtype = data_compact.dtype

    if not less_output:
        print('llc_compact_to_faces: dims, llc ', dims, llc)
        print('llc_compact_to_faces: data_compact array type ', data_compact.dtype)

    if num_dims == 2: # we have a single 2D slices (y, x)
        f1 = np.zeros((3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((llc, llc), dtype=arr_dtype)
        f4 = np.zeros((llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((llc, 3*llc), dtype=arr_dtype)

    elif num_dims == 3: # we have 3D slices (time or depth, y, x)
        nk = dims[0]

        f1 = np.zeros((nk, 3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nk, 3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nk, llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((nk, llc, 3*llc), dtype=arr_dtype)

    elif num_dims == 4: # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        f1 = np.zeros((nl, nk, 3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nl, nk, 3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nl, nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nl, nk, llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((nl, nk, llc, 3*llc), dtype=arr_dtype)

    else:
        print ('llc_compact_to_faces: can only handle compact arrays of 2, 3, or 4 dimensions!')
        return []

    # map the data from the compact format to the five face arrays

    # -- 2D case
    if num_dims == 2:

        f1 = data_compact[:3*llc,:]
        f2 = data_compact[3*llc:6*llc,:]
        f3 = data_compact[6*llc:7*llc,:]

        #f4 = np.zeros((llc, 3*llc))

        for f in range(8,11):
            i1 = np.arange(0, llc)+(f-8)*llc
            i2 = np.arange(0,3*llc,3) + 7*llc + f -8
            f4[:,i1] = data_compact[i2,:]

        #f5 = np.zeros((llc, 3*llc))

        for f in range(11,14):
            i1 = np.arange(0, llc)+(f-11)*llc
            i2 = np.arange(0,3*llc,3) + 10*llc + f -11
            #print ('f, i1, i2 ', f, i1[0], i2[0])

            f5[:,i1] = data_compact[i2,:]

    # -- 3D case
    elif num_dims == 3:
        # loop over k

        for k in range(nk):
            f1[k,:] = data_compact[k,:3*llc,:]
            f2[k,:] = data_compact[k,3*llc:6*llc,:]
            f3[k,:] = data_compact[k,6*llc:7*llc,:]

            # if someone could explain why I have to make
            # dummy arrays of f4_tmp and f5_tmp instead of just using
            # f5 directly I would be so grateful!
            f4_tmp = np.zeros((llc, 3*llc))
            f5_tmp = np.zeros((llc, 3*llc))

            for f in range(8,11):
                i1 = np.arange(0, llc)+(f-8)*llc
                i2 = np.arange(0,3*llc,3) + 7*llc + f -8
                f4_tmp[:,i1] = data_compact[k,i2,:]


            for f in range(11,14):
                i1 = np.arange(0,  llc)   +(f-11)*llc
                i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                f5_tmp[:,i1] = data_compact[k,i2,:]

            f4[k,:] = f4_tmp
            f5[k,:] = f5_tmp



    # -- 4D case
    elif num_dims == 4:
        # loop over l and k
        for l in range(nl):
            for k in range(nk):

                f1[l,k,:] = data_compact[l,k,:3*llc,:]
                f2[l,k,:] = data_compact[l,k, 3*llc:6*llc,:]
                f3[l,k,:] = data_compact[l,k, 6*llc:7*llc,:]

                # if someone could explain why I have to make
                # dummy arrays of f4_tmp and f5_tmp instead of just using
                # f5 directly I would be so grateful!
                f4_tmp = np.zeros((llc, 3*llc))
                f5_tmp = np.zeros((llc, 3*llc))

                for f in range(8,11):
                    i1 = np.arange(0, llc)+(f-8)*llc
                    i2 = np.arange(0,3*llc,3) + 7*llc + f -8
                    f4_tmp[:,i1] = data_compact[l,k,i2,:]

                for f in range(11,14):
                    i1 = np.arange(0, llc)+(f-11)*llc
                    i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                    f5_tmp[:,i1] = data_compact[l,k,i2,:]

                f4[l,k,:,:] = f4_tmp
                f5[l,k,:,:] = f5_tmp


    # put the 5 faces in the dictionary.
    F = {}
    F[1] = f1
    F[2] = f2
    F[3] = f3
    F[4] = f4
    F[5] = f5

    return F

def llc_faces_to_tiles(F, less_output=False):
    """

    Converts a dictionary, F, containing 5 lat-lon-cap faces into 13 tiles
    of dimension nl x nk x llc x llc x nk.

    Tiles 1-6 and 8-13 are oriented approximately lat-lon
    while tile 7 is the Arctic 'cap'

    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_tiles : ndarray
        an array of dimension 13 x nl x nk x llc x llc,

        Each 2D slice is dimension 13 x llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_tiles


    """

    # pull out the five face arrays
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    dims = f3.shape
    num_dims = len(dims)

    # dtype of compact array
    arr_dtype = f1.dtype

    # final dimension of face 1 is always of length llc
    ni_3 = f3.shape[-1]

    llc = ni_3 # default
    #

    if num_dims == 2: # we have a single 2D slices (y, x)
        data_tiles = np.zeros((13, llc, llc), dtype=arr_dtype)


    elif num_dims == 3: # we have 3D slices (time or depth, y, x)
        nk = dims[0]
        data_tiles = np.zeros((nk, 13, llc, llc), dtype=arr_dtype)


    elif num_dims == 4: # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        data_tiles = np.zeros((nl, nk, 13, llc, llc), dtype=arr_dtype)

    else:
        print ('llc_faces_to_tiles: can only handle face arrays that have 2, 3, or 4 dimensions!')
        return []

    # llc is the length of the second dimension
    if not less_output:
        print ('llc_faces_to_tiles: data_tiles shape ', data_tiles.shape)
        print ('llc_faces_to_tiles: data_tiles dtype ', data_tiles.dtype)


    # map the data from the faces format to the 13 tile arrays

    # -- 2D case
    if num_dims == 2:
        data_tiles[0,:]  = f1[llc*0:llc*1,:]
        data_tiles[1,:]  = f1[llc*1:llc*2,:]
        data_tiles[2,:]  = f1[llc*2:,:]
        data_tiles[3,:]  = f2[llc*0:llc*1,:]
        data_tiles[4,:]  = f2[llc*1:llc*2,:]
        data_tiles[5,:]  = f2[llc*2:,:]
        data_tiles[6,:]  = f3
        data_tiles[7,:]  = f4[:,llc*0:llc*1]
        data_tiles[8,:]  = f4[:,llc*1:llc*2]
        data_tiles[9,:]  = f4[:,llc*2:]
        data_tiles[10,:] = f5[:,llc*0:llc*1]
        data_tiles[11,:] = f5[:,llc*1:llc*2]
        data_tiles[12,:] = f5[:,llc*2:]

    # -- 3D case
    if num_dims == 3:
        # loop over k
        for k in range(nk):
            data_tiles[k,0,:]  = f1[k,llc*0:llc*1,:]
            data_tiles[k,1,:]  = f1[k,llc*1:llc*2,:]
            data_tiles[k,2,:]  = f1[k,llc*2:,:]
            data_tiles[k,3,:]  = f2[k,llc*0:llc*1,:]
            data_tiles[k,4,:]  = f2[k,llc*1:llc*2,:]
            data_tiles[k,5,:]  = f2[k,llc*2:,:]
            data_tiles[k,6,:]  = f3[k,:]
            data_tiles[k,7,:]  = f4[k,:,llc*0:llc*1]
            data_tiles[k,8,:]  = f4[k,:,llc*1:llc*2]
            data_tiles[k,9,:]  = f4[k,:,llc*2:]
            data_tiles[k,10,:] = f5[k,:,llc*0:llc*1]
            data_tiles[k,11,:] = f5[k,:,llc*1:llc*2]
            data_tiles[k,12,:] = f5[k,:,llc*2:]

    # -- 4D case
    if num_dims == 4:
        #loop over l and k
        for l in range(nl):
            for k in range(nk):
                data_tiles[l,k,0,:]  = f1[l,k,llc*0:llc*1,:]
                data_tiles[l,k,1,:]  = f1[l,k,llc*1:llc*2,:]
                data_tiles[l,k,2,:]  = f1[l,k,llc*2:,:]
                data_tiles[l,k,3,:]  = f2[l,k,llc*0:llc*1,:]
                data_tiles[l,k,4,:]  = f2[l,k,llc*1:llc*2,:]
                data_tiles[l,k,5,:]  = f2[l,k,llc*2:,:]
                data_tiles[l,k,6,:]  = f3[l,k,:]
                data_tiles[l,k,7,:]  = f4[l,k,:,llc*0:llc*1]
                data_tiles[l,k,8,:]  = f4[l,k,:,llc*1:llc*2]
                data_tiles[l,k,9,:]  = f4[l,k,:,llc*2:]
                data_tiles[l,k,10,:] = f5[l,k,:,llc*0:llc*1]
                data_tiles[l,k,11,:] = f5[l,k,:,llc*1:llc*2]
                data_tiles[l,k,12,:] = f5[l,k,:,llc*2:]

    return data_tiles

def llc_compact_to_tiles(data_compact, less_output = False):
    """

    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.

    Parameters
    ----------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output


    Returns
    -------
    data_tiles : ndarray
        a numpy array organized by, at most,
        13 tiles x nl x nk x llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions in data_tiles

    """

    data_tiles =  llc_faces_to_tiles(
                    llc_compact_to_faces(data_compact,
                                         less_output=less_output),
                    less_output=less_output)

    return data_tiles