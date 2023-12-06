from artifactSimulation import spatial, sample
import random
import numpy
import pandas
from pathlib import Path


def machine_spots(positions):
    """
    selects barcodes for inducing machine artifacts
    First it chooses a random spot with a minimum distance of 3
    from any edge (tissue or capture border).
    Next it adds all of it's neighbors, then the neighbors of one of
    it's neighbors

    parameters:
    ---------
        pos: a pandas Dataframe from a file like tissue_positions.csv
            with barcodes set as the index

    returns: a list of barcodes
    """
    generator = numpy.random.default_rng()
    pos = positions.copy()
    pos['t_dist'] = spatial.tissue_edge_distance(pos)
    pos['c_dist'] = spatial.capture_edge_distance(pos)
    pos['distance'] = pos[['t_dist', 'c_dist']].min(axis=1)
    spot = generator.choice(pos[pos.distance >= 3].index)
    print('Start spot:', spot)
    neighbors = spatial.neighbors(pos, spot)
    next_spot1, next_spot2 = generator.choice(neighbors, size=2)
    neighbors += spatial.neighbors(pos, next_spot1)
    neighbors += spatial.neighbors(pos, next_spot2)
    return list(set(neighbors))


def induce_machine_artifacts(df, pos, spots=None):
    """
    alter a dataframe to have a lump of machine artifacts

    paramters:
    ---------
        df: a pandas Dataframe from a 10x Slide
            the index is barcodes
            the columns are gene aliases
        pos: a pandas Dataframe from a file like tissue_positions.csv
        spots: a list of barcodes of spots to adjust

    returns:
    -------
        df: the dataframe with the edge and border spots removed
        pos: the positions dataframe with the in_tissue field updated
    """
    factor = 0.002074695

    if spots is None:
        spots = machine_spots(pos)
    df.loc[spots] *= factor
    return df


def strip_sample(df, pos):
    """
    removes the outer edges of a Visium sample, both the capture border
    and the tissue edge; both of which tend to be artifacted.

    paramters:
    ---------
        df: a pandas Dataframe from a 10x Slide
            the index is barcodes
            the columns are gene aliases
        pos: a pandas Dataframe from a file like tissue_positions.csv

    returns:
    -------
        df: the dataframe with the edge and border spots removed
        pos: the positions dataframe with the in_tissue field updated
    """
    tissue_edge_distances = spatial.tissue_edge_distance(pos)
    capture_edge_distances = spatial.capture_edge_distance(pos)

    if len(capture_edge_distances[capture_edge_distances == 1]) > 0:
        df_cap_stripped = spatial.strip_border(df, pos)
        final_index = set(df_cap_stripped.index)
    else:
        final_index = set(df.index)

    if len(tissue_edge_distances[tissue_edge_distances == 1]) > 0:
        df_edge_stripped = spatial.strip_edge(df, pos)
        edge_index = set(df_edge_stripped.index)
        final_index = list(final_index.intersection(edge_index))

    df = df.loc[final_index]
    pos = spatial.update_positions(pos, df)

    return df, pos


def make_fake_sample(df, pos):
    """ alters a 10x sample by removing it's capture edge and tissue edges
    and then shifts the sample horizontally and vertically so that it has
    a capture edge. Also updates the positions dataframe.

    # TODO: add in catches for when tissue sample is too small

    parameters:
    ---------
        df: a pandas Dataframe from a 10X visium sample
            (hint: use the sample.py file)
        pos: a pandas Dataframe from a file like tissue_positions.csv
            with barcodes set as the index

    returns: the altered df and pos (as a pair)
    """
    df, pos = strip_sample(df, pos)
    vert_shift = random.choice([-2, 2])
    horiz_shift = random.choice([-2, 2])
    border_min = 10

    if len(df) > 1000:
        capture_edge = spatial.capture_edge_distance(pos)
        while len(capture_edge[capture_edge == 1]) < border_min:
            print('Edge spots:', len(capture_edge[capture_edge == 1]))
            print('shifting up')
            df = spatial.shift_up(df, pos, vert_shift)
            pos = spatial.update_positions(pos, df)
            capture_edge = spatial.capture_edge_distance(pos)
        border_min += len(capture_edge[capture_edge == 1])
        while len(capture_edge[capture_edge == 1]) <= border_min:
            df = spatial.shift_right(df, pos, horiz_shift)
            pos = spatial.update_positions(pos, df)
            capture_edge = spatial.capture_edge_distance(pos)
    return df, pos


def high_frequency_gene_table(topdir, sample_genes=200):
    """ create a table of gene frequencies from a directory of samples.
    It is assumed that each subdirectory of `topdir` is a 10X sample directory.

    paramters:
    ---------
        topdir: a Path of a string of the path to a directory containing 10X
            sample directories
        sample_genes: the number of genes to select from each sample
            genes are selected by appearing in the highest number of
            spots in each sample

    returns:
    -------
        a DataFrame with the names of the 
    """
    topdir = Path(topdir)
    frequencies = dict()

    # Loop through samples and collect a dictionaries of the highest
    # frequency genes in each sample
    for sampledir in topdir.iterdir():
        s = sample.Sample(sampledir)
        df = s.df.copy()
        df[df > 0] = 1
        top_genes = df.sum().sort_values(ascending=False).head(sample_genes)
        freqs = {k: df[k].sum() / len(df) for k in list(top_genes)}
        frequencies[sampledir.name] = freqs

    gene_freqs = pandas.DataFrame(frequencies)


def tissue_summary(name, status, pos):
    """ Not sure why this is here
    """
    tissue_edge_distances = spatial.tissue_edge_distance(pos)
    capture_edge_distances = spatial.capture_edge_distance(pos)
    summary = dict()
    summary['sample'] = name
    summary['status'] = status
    summary['spots'] = len(pos[pos.in_tissue == 1])
    summary['border_one'] = len(capture_edge_distances[capture_edge_distances == 1])
    summary['border_two'] = len(capture_edge_distances[capture_edge_distances == 2])
    summary['edge_one'] = len(tissue_edge_distances[tissue_edge_distances == 1])
    summary['edge_two'] = len(tissue_edge_distances[tissue_edge_distances == 2])

    return summary
