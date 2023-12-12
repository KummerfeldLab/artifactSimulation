import pandas
import scanpy
from artifactSimulation import spatial
from pathlib import Path


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
        new_s_name = 'NewSample.hd5'
        hd5_name = 'filtered_feature_bc_matrix.mtx.gz'
        mtx_name = 'matrix.mtx.gz'

        if sampledir.joinpath(hd5_name).exists():
            self.ann = scanpy.read_10x_h5(sampledir / hd5_name)
        elif (sampledir / new_s_name).exists():
            self.ann = scanpy.read_10x_h5(sampledir / new_s_name)
        elif (sampledir / mtx_name).exists():
            self.ann = scanpy.read_10x_mtx(sampledir, cache=True)
        else:
            raise ValueError('No 10X data found at ' + str(sampledir))
        self.df = self.ann.to_df()

        positions_source = sampledir
        if sampledir.joinpath('spatial').exists():
            positions_source = sampledir / 'spatial'

        if (positions_source / old_postions_f).exists():
            positions_source = positions_source / old_postions_f
        elif (positions_source / positions_f).exists():
            positions_source = positions_source / positions_f

        self.positionsdf = pandas.read_csv(positions_source, engine='pyarrow')
        self.positionsdf.set_index('barcode', inplace=True)
        self.positionsdf.sort_values(['array_col', 'array_row'], inplace=True)

    def tissue_edge_distance(self):
        """provides a pandas.Series of tissue edge distances of the spots"""
        return spatial.tissue_edge_distance(self.positionsdf)

    def get_barcode_neighbors(self, barcode):
        """ returns a list of all neighbors of a barcode according
        to self.positionsdf """
        return spatial.neighbors(self.positionsdf, barcode)
