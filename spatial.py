import pandas
import h5py
from scipy.sparse import csr_matrix
from scanpy import read_10x_h5


def neighbors(pos, barcode):
    """
    neighbors(pos, barcode) returns the neighbors of a spot in a
    10x visium scan id'd by a barcode

    relies on the position dataframe's in_tissue flag
    TODO: change the reliance on the in_tissue flag
        so to be more generic
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
    return len(neighbors(pos, barcode))


def border(pos):
    border_cols = [pos.array_col.min(),
                   pos.array_col.min() + 1,
                   pos.array_col.max() - 1,
                   pos.array_col.max()]
    border_rows = [pos.array_row.min(), pos.array_row.max()]
    return pos.loc[(pos.array_col.isin(border_cols) |
                    pos.array_row.isin(border_rows)) &
                   (pos.in_tissue == 1)].index


def capture_edge_distance(pos):
    return get_border_distance(pos)


def tissue_edge_distance(pos):
    return edge_distance(pos)


def get_border_distance(pos):
    bd = pandas.Series(index=pos.index, dtype='Int32')
    cur_dist = 1
    bd[border(pos)] = cur_dist
    while len(bd[bd.isna()]) > 0:
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


def edge_distance(pos):
    """ provides a Series of the distances of each spot
            from the edge of the tissue.
        The Series index are the spot barcodes
        If the spot lies on the capture border, it will have distance 0.
        The tissue edge is determined by the in_tissue flag of pos.
        If pos is not current to the barcodes, then the resultant
        data will not be consistant.
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


def read_hd5_dir(path):
    pos = pandas.read_csv(path / 'spatial/tissue_positions.csv')
    s = read_10x_h5(path / 'filtered_feature_bc_matrix.h5')
    return s, pos.set_index('barcode')


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

    bd = get_border_distance(pos)
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
