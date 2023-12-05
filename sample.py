import numpy
import os
import pandas
import scanpy
from SNPipeline.genelists import genelists
from SNPipeline import spatial
from pathlib import Path
from ckmeans_1d_dp import ckmeans


class Sample:
    def __init__(self, dirname):
        """
            reads in 10X Visium and CytAssist data for artifact simulation
            The initial data is read into self.ann as an AnnData object
            No preprocessing is performed.
            The data is then held in self.df as a pandas DataFrame.
            The barcode positions table is then read from dirname/spatial/
            Can read from either the oldstile tissue_positions_data.to_csv
            Or the current tissue_positions.csv.
            (Older versions of scanpy did not support the newer filename)

        """
        old_postions_f = 'tissue_positions.csv'
        positions_f = 'tissue_positions.csv'
        sampledir = Path(dirname)
        self.sample = sampledir.name
        msi_name = 'filtered_feature_bc_matrix'
        positions_source = Path(dirname) / positions_f
        if sampledir.joinpath(msi_name).exists():
            sampledir = sampledir.joinpath(msi_name)
            positions_source = Path(dirname).joinpath('spatial')
            if positions_source.joinpath(positions_f)
                positions_source = positions_source.joinpath(positions_f)
            else:
                positions_source = positions_source.joinpath(old_postions_f)
        self.ann = scanpy.read_10x_mtx(sampledir, cache=True)
        self.df = self.ann.to_df()

        if genelist is not None:
            self.df = self.df[genelist]

        self.positionsdf = pandas.read_csv(positions_source, engine='pyarrow')
        self.positionsdf.set_index('barcode', inplace=True)
        self.positionsdf.sort_values(['array_col', 'array_row'], inplace=True)

    def get_edge_distance(self):
        edges = pandas.Series(index=self.positionsdf.index, dtype='float64')
        edges[edges.isna()] = numpy.inf
        zero_idx = self.positionsdf[self.positionsdf.in_tissue == 0].index
        edges[zero_idx] = 0

        cur_dist = 0
        inf_edges = edges[numpy.isinf(edges)].index
        while len(inf_edges) > 0:
            neighbors = set()
            cur_index = edges[edges == cur_dist].index
            for barcode in cur_index:
                neighbors.update(self.get_barcode_neighbors(barcode))

            neighbors.intersection_update(inf_edges)
            edges[list(neighbors)] = cur_dist + 1
            inf_edges = edges[numpy.isinf(edges)].index
            cur_dist += 1
        return edges.astype('int32')

    def get_barcode_neighbors(self, barcode):
        """ returns a list of all neighbors of a barcode according
        to self.positionsdf """

        return spatial.neighbors(self.positionsdf, barcode)

    def get_neighbors(self, df, r, c):
        """ get_neighbors() is a utilty function used by
        get_adjacency_matrix """

        n_list = list()
        neighbor = df.loc[(df.array_col == c + 1) & (df.array_row == r)]
        if len(neighbor) == 1:
            n_list.append(neighbor.index.min())
        neighbor = df.loc[(df.array_col == c + 1) & (df.array_row == r + 1)]
        if len(neighbor) == 1:
            n_list.append(neighbor.index.min())
        neighbor = df.loc[(df.array_col == c + 1) & (df.array_row == r - 1)]
        if len(neighbor) == 1:
            n_list.append(neighbor.index.min())

        return n_list

    def build_adjacency_matrix(self):
        """ Used by get_adjacency_matrix to build adjacency matrix
        if it does not exist """
        positions = self.get_positions()
        barcodes = positions.index.to_list()
        s = {bc: numpy.zeros(shape=len(barcodes)) for bc in barcodes}
        df = pandas.DataFrame(s, index=barcodes)
        for bc in barcodes:
            col = positions.at[bc, 'array_col']
            row = positions.at[bc, 'array_row']

            for neighbor in self.get_neighbors(self.positionsdf, row, col):
                df.loc[bc, neighbor] = 1
                df.loc[neighbor, bc] = 1

        return df

    def get_adjacency_matrix(self):
        """ Reads in the adjacency matrix and trims it
        to be specific to this tissue sample """
        adj_fname = 'adjacencyMatrix.csv.gz'
        adj_path = self.dirname.parent.joinpath(adj_fname)
        if adj_fname not in os.listdir(self.dirname.parent):
            M = self.build_adjacency_matrix()
            M.to_csv(adj_path, compression='gzip', index=False)
        else:
            M = pandas.read_csv(adj_path, compression='gzip')
        M.set_index(keys=M.columns, inplace=True)
        M = M.loc[self.positionsdf.index, self.positionsdf.index]
        return M

    def get_genelist_population(self, gl, add_positions=True):
        genes = genelists[gl]
        genelist_df = pandas.DataFrame()
        for gene in [x for x in genes if x in self.df.columns]:
            genelist_df[gene] = self.df[gene]
        genelist_df['Listgenes'] = genelist_df.sum(axis=1)
        genelist_df['All'] = self.df.sum(axis=1)

        numerator = genelist_df['Listgenes'].sum()
        expected_ratio = numerator / genelist_df['All'].sum()

        genelist_df['Expected'] = genelist_df['All'].mul(expected_ratio)
        if add_positions:
            genelist_df = genelist_df.merge(self.positionsdf,
                                            left_index=True,
                                            right_index=True)

        return genelist_df

    def permute(self, nperms=200):
        df = self.df.copy()
        ed = self.edgeDistance
        maxdist = ed.max()
        barcodes = {x: ed[ed == x].index for x in range(1, maxdist)}
        newindex = df.index.values
        m = []
        s = []
        for _ in range(nperms):
            numpy.random.shuffle(newindex)
            df = df.set_index(newindex)
            drange = range(1, ed.max())
            means = {i: df.loc[barcodes[i]].sum(axis=1).mean() for i in drange}
            drange = range(1, ed.max())
            stds = {i: df.loc[barcodes[i]].sum(axis=1).std() for i in drange}
            m.append(pandas.Series(means))
            s.append(pandas.Series(stds))
        return pandas.concat(m, axis=1), pandas.concat(s, axis=1)
