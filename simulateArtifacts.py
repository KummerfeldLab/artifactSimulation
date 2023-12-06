from SNPipeline import spatial
import random
import numpy


def machine_spots(positions):
    """
    selects barcodes for inducing machine artifacts

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


def induce_mean_shift(pos, df, distance, edgetype='capture', excludes=[]):
    """

    paramters:
    ---------
        pos: a pandas Dataframe from a file like tissue_positions.csv
        df: a pandas Dataframe from a 10x Slide
            the index is barcodes
            the columns are gene aliases
        distance: the distance from the edge that should be induced
        edgetype: whether to induce capture border spots,
                  or rather tissue edge spots
        excludes: a set of barcodes not to induce, even if they would
                    otherwise meet edgetype and distance criteria

    """
    TISSUE_MEANS = (0.9661, 1.1028)
    CAPTURE_MEANS = (1.4504, 1.0499)

    if distance < 1 or distance > 2:
        raise ValueError("Distance must be 1 or 2")

    if edgetype == 'capture':
        ed = spatial.capture_edge_distance(pos)
        induction_mean = CAPTURE_MEANS[distance - 1]
    elif edgetype == 'tissue':
        ed = spatial.tissue_edge_distance(pos)
        induction_mean = TISSUE_MEANS[distance - 1]
    else:
        raise ValueError("edgetype must be `capture` or `tissue`")

    barcodes = [x for x in ed[ed == distance].index if x not in excludes]
    df.loc[barcodes] = df.loc[barcodes].mul(induction_mean)
    df = df.round()

    return df, list(barcodes), list(barcodes) + excludes


def make_fake_sample(df, pos):
    # TODO: add in catches for when tissue sample is too small
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


def tissue_summary(name, status, pos):
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
