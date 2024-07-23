import pandas
import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import ttest_ind


def neighbors(pos, barcode):
    """
    returns the neighbors of a spot in a 10x visium scan id'd by a barcode
    only provides neighboring spots where there is tissue

    parameters:
    ---------
        pos: a pandas Dataframe created by reading from tissue_positions.csv
            with the index set to barcodes
        barcode: a barcode appearing in pos

    returns:
    ------
        a list of neighbors of a spot
    """
    r = pos.at[barcode, 'array_row']
    c = pos.at[barcode, 'array_col']
    cols = [c-2, c-1, c, c+1, c+2]
    rows = [r-1, r, r+1]
    bc_l = pos.loc[pos.array_row.isin(rows) &
                   pos.array_col.isin(cols) &
                   (pos.in_tissue == 1)].index
    bc_l = bc_l.to_list()
    if barcode in bc_l:
        bc_l.remove(barcode)
    return bc_l


def num_neighbors(pos, barcode):
    """ provides the number of in_tissue neighbors of a spot """
    return len(neighbors(pos, barcode))


def senspot_index(sample, genelist):
    """
    Identifies senspots according to a given list and returns
    the barcodes of senspots.
    A spot is considered senescent if it's mean
    zscore in the provided dataframe is above 2
    """
    s1 = sample[genelist].sum(axis=1).div(len(genelist))
    result = list(s1[s1 > 2].index)
    return result


def spot_distance(tissue_positions, senspots, maxdist=6):
    d = pandas.Series(np.inf, index=tissue_positions.index)
    d[senspots] = 0
    cur_d = 0
    while (len(d[np.isinf(d)]) > 0) and cur_d <= maxdist:
        spot_neighbors = set()
        inners = d[d >= cur_d].index
        cur_positions = tissue_positions.loc[inners]
        for spot in d[d == cur_d].index:
            spot_neighbors.update(neighbors(cur_positions, spot))
        spot_neighbors = list(spot_neighbors)
        d.loc[(d > cur_d) & (d.index.isin(spot_neighbors))] = cur_d + 1
        cur_d += 1

    return d


def border(pos):
    """ Provides the barcodes of spots that are on the capture border
            but only the ones with tissue present.

    parameters:
    ----------
        pos: a pandas Dataframe created by reading from tissue_positions.csv
            with the index set to barcodes

    returns:
    ------
        a list of the barcodes of capture edge spots
    """
    border_cols = [pos.array_col.min(),
                   pos.array_col.min() + 1,
                   pos.array_col.max() - 1,
                   pos.array_col.max()]
    border_rows = [pos.array_row.min(), pos.array_row.max()]
    border = pos.loc[(pos.array_col.isin(border_cols) |
                      pos.array_row.isin(border_rows)) &
                     (pos.in_tissue == 1)].index
    return list(border)


def capture_edge_distance(pos, width=np.nan):
    """ provides the distance of each spot up to width -width-
        from the edge of capture window
        the outmost in-tissue spots are assigned a distance value of one

    parameters:
    ----------
        pos: a pandas Dataframe created by reading from tissue_positions.csv
            with the index set to barcodes
        width: the maximum distance from the edge to consider
            restricting width improves processing time

    returns:
    ------
        a pandas Series of the distances, indexed by barcodes
    """
    cur_dist = 1
    bd = pandas.Series(index=pos.index, dtype='Int32')
    bd[border(pos)] = cur_dist
    while len(bd[bd.isna()]) > 0 and cur_dist <= width:
        n = set()
        for barcode in bd[bd == cur_dist].index:
            n.update(neighbors(pos, barcode))
        n.intersection_update(bd[bd.isna()].index)
        n = list(n)
        if len(n) == 0:
            break
        # filter for those that are not nan
        bd[list(n)] = cur_dist + 1
        cur_dist += 1
    bd[bd.isna()] = 0
    return bd


def tissue_edge_distance(pos):
    """ provides the distances of each spot from the edge of the tissue.
        Spots that do not contain tissue are assigned distance 0.
        Neighbors of spots with missing tissue that do contain tissue
        are assigned distance 1.

    parameters:
    ----------
        pos: a pandas Dataframe created by reading from tissue_positions.csv
            with the index set to barcodes

    returns:
    ------
        a pandas Series of the distances, indexed by barcodes
    """

    edges = pandas.Series(index=pos.index, dtype='int32[pyarrow]')
    edges[edges.isna()] = pandas.NA
    zero_idx = pos[pos.in_tissue == 0].index
    edges[zero_idx] = 0

    cur_dist = 0
    na_spots = edges[edges.isna()].index
    while len(na_spots) > 0:
        layer_neighbors = set()
        cur_index = edges[edges == cur_dist].index
        for barcode in cur_index:
            layer_neighbors.update(neighbors(pos, barcode))

        layer_neighbors.intersection_update(na_spots)
        edges[list(layer_neighbors)] = cur_dist + 1
        na_spots = edges[edges.isna()].index
        cur_dist += 1
    return edges


def strip_border(data, pos, width=2):
    """ removes the rows of data corrisponding to barcodes
        on the capture border, up do distance width from the
        border

        parameters:
        ----------
            data: a pandas Series or Dataframe indexed by barcodes
            pos: a pandas Dataframe from tissue_positions.csv
            width: the depth into which barcodes should be removed
    """

    bd = capture_edge_distance(pos, width)
    data = data.loc[bd[bd > width].index]
    return data


def strip_edge(data, pos, width=2):
    """ removes the rows of data corrisponding to barcodes
        on the edge of the tissue, up do distance width from the
        border
        Warning: depends on the in_tissue flag of positions.
        The positions data must be updated seperately to this function.

        parameters:
        ----------
            data: a pandas Series or Dataframe indexed by barcodes
            pos: a pandas Dataframe from tissue_positions.csv
            width: the depth into which barcodes should be removed
    """
    edgedist = tissue_edge_distance(pos)
    return data.loc[edgedist[edgedist > width].index]


def shift_right(data, pos, distance=2):
    if distance % 2 != 0:
        raise ValueError('distance must be an even number')
    if not type(data) in [pandas.Series, pandas.DataFrame]:
        raise ValueError('data must be a pandas Series or DataFrame')

    on_cols = ['array_row', 'array_col']
    altered_pos = pos.loc[data.index, on_cols].copy()
    altered_pos['array_row'] = altered_pos['array_row'].sub(distance)
    altered_pos.reset_index(inplace=True)
    altered_pos.set_index(on_cols, inplace=True)
    new_idx = pos.join(altered_pos, on=on_cols, how='right').index
    if type(data) is pandas.Series:
        new_data = pandas.Series(data.values, index=new_idx)
    else:
        new_data = data.set_index(keys=new_idx, drop=True)
    return new_data


def shift_up(data, pos, distance=2):
    if distance % 2 != 0:
        raise ValueError('distance must be an even number')
    if not type(data) in [pandas.Series, pandas.DataFrame]:
        raise ValueError('data must be a pandas Series or DataFrame')

    on_cols = ['array_row', 'array_col']
    altered_pos = pos.loc[data.index, on_cols].copy()
    altered_pos['array_col'] = altered_pos['array_col'].sub(distance)
    altered_pos.reset_index(inplace=True)
    altered_pos.set_index(on_cols, inplace=True)
    new_idx = pos.join(altered_pos, on=on_cols, how='right').index
    if type(data) is pandas.Series:
        new_data = pandas.Series(data.values, index=new_idx)
    else:
        new_data = data.set_index(keys=new_idx, drop=True)
    return new_data


def test_and_strip(pos, df, threshold=0.05, max_distance=10):
    """ given a positions dataframe similar to tissue_positions.csv,
        and a dataframe from a spaceranger-processed 10X Visium result
        test border distances and edge distances for having an abnormal
        distribution, and return the barcodes of spots that do """

    excludes = set()
    border_dist = capture_edge_distance(pos)
    edge_dist = tissue_edge_distance(pos)
    for d in range(1, max_distance + 1):
        test_spots = border_dist[border_dist == d].index
        new_test_spots = list(set(test_spots).difference(excludes))
        interior_spots = border_dist[border_dist > d].index
        ttest = ttest_ind(df.loc[new_test_spots].sum(axis=1),
                          df.loc[interior_spots].sum(axis=1))
        pvalue = ttest.pvalue
        display_str = "At Border Distance = {}, the p_value is {:.4f}"
        print(display_str.format(d, pvalue), end=" ")
        if pvalue > threshold:
            print('\tPASS')
            break
        print('\tFAIL')
        excludes.update(set(new_test_spots))
        
    for d in range(1, max_distance + 1):
        test_spots = edge_dist[edge_dist == d].index
#       new_test_spots = list(set(test_spots).difference(excludes))
        new_test_spots = test_spots
        interior_spots = edge_dist[edge_dist > d].index
        ttest = ttest_ind(df.loc[new_test_spots].sum(axis=1),
                          df.loc[interior_spots].sum(axis=1))
        pvalue = ttest.pvalue
        display_str = "At Edge Distance = {}, the p_value is {:.4f}"
        print(display_str.format(d, pvalue), end=" ")
        if pvalue > threshold:
            print('\tPASS')
            break
        print('\tFAIL')
        excludes.update(set(new_test_spots))
        
    return excludes




def update_positions(pos, df):
    pos.in_tissue = 0
    pos.loc[df.index, 'in_tissue'] = 1
    return pos


def write_hdf5(filename, df, ensembl_ids, genome_name='GRCh38'):
    """ writes an hdf5 in the format of visium files
    See https://support.10xgenomics.com/spatial-gene-expression/software/
    pipelines/latest/advanced/h5_matrices

    parameters:
    ----------
        filename (string): the name of the file to write to
            note that the file extension must be given here.
            It will not be added by this method.

        df (pandas.DataFrame): a table of gene counts indexed by barcodes

        ensembl_ids (string collection): an iterable of strings of ENSEMBL
            codes corresponding to each gene in df
             These can either be read from the features.tsv.gz file
             -or-
             found in the ann returned by read_10_hd5 or similar
             using list(ann.var['gene_ids'])


        genome_name (string): an interable of strings that gives the
            genome for each corresponding ensembl_id.
            TODO: handle multiple genomes

    """
    with h5py.File(filename, 'w') as f:

        # Create `matrix`, which serves as  the top-level directory
        # of the visium data lump
        mat = f.create_group('matrix')
        mat.create_dataset('barcodes', data=list(df.index),
                           dtype=h5py.string_dtype(length=18))

        # Create the features group to contain the list of genes
        # and related fields
        features = mat.create_group('features')
        # Add in the single string that makes up the _all_tag_keys data set.
        # Note: h5py doesn't automatically convert strings,
        # they must be specified to be strings.
        # the string must be inside as 10x related utilities expect a list
        features.create_dataset('_all_tag_keys', data=['genome'],
                                dtype=h5py.string_dtype(length=8))

        # The feature_type data set is just a list of copies of the string
        #   `Gene Expression` once for every gene in the dataset.
        # Since genes are the columns of the dataset, we use the number of
        #   columns as the number of genes
        num_genes = len(df.columns)
        datastr = 'Gene Expression'
        features.create_dataset('feature_type',
                                data=[datastr] * num_genes,
                                dtype=h5py.string_dtype(length=len(datastr)))

        # The genome dataset is genome_name repeated num_genes times.
        # The default is GRCh38, which the the designation for human genomes
        strlen = len(genome_name)
        features.create_dataset('genome',
                                data=[genome_name] * num_genes,
                                dtype=h5py.string_dtype(length=strlen))

        # name is the actually useful bit, the list of gene names
        gene_names = list(df.columns)
        features.create_dataset('name',
                                data=gene_names,
                                dtype=h5py.string_dtype(length=11))

        # ids is the ENSEMBLE codes for each name
        # These can either be
        # read from the features.tsv.gz file
        # -or-
        # found in the ann returned by read_10_hd5 or similar
        # using list(ann.var['gene_ids'])

        # and grab the ENSEMBL codes corresponding to GeneName
        # HD5 does not recognize pandas Series, so it must be made into a list
        # TODO: make sure ENSEMBL ids are in the correct order
#       ensembl_ids = featuresdf.loc[featuresdf.GeneName.isin(gene_names),
#                                    'Ensembl_id'].to_list()
        features.create_dataset('id',
                                data=ensembl_ids,
                                dtype=h5py.string_dtype(length=15))

        # The other datasets in the HD5 come from the counts and properties of
        # a sparse count matrix
        # First we create a compressed sparse column matrix of the liver_df
        # The properties of the datasets are easily found in the csc_matrix
        M = csr_matrix(df.values)
        c, r = M.shape
        mat.create_dataset('shape', data=[r, c])
    #    mat.create_dataset('shape', data=M.shapeA
        mat.create_dataset('indices', data=M.indices)
        mat.create_dataset('indptr', data=M.indptr)
        mat.create_dataset('data', data=M.data, dtype='i')
