import pandas
from simulateArtifacts import sample, spatial
from scipy.stats import ttest_ind, kstest, mannwhitneyu
from scipy.stats import false_discovery_control
from pathlib import Path
from numpy import isnan


def multisample_border_stats(topdir, btype='Capture'):
    """ Compiles summary stats and pvalues of means comparison tests of
        all 10X samples in a directory (folder)
        The comparison is either Capture Edge spots vs interior spots,
        or tissue edge spots vs interior spots
        The comparison considers several 'distances' to evaluate how 'deep'
        the effect is.
        If you want tables for both capture edge and tissue edge, run twice,
        changing the btype parameter

    parameters:
    ----------
        p: a pandas Dataframe from 10X tissue_positions.csv
        btype: the type of artifact {'Capture', 'Tissue'} to compare

    returns:
    -------
        A table with columns the name of the sample and rows the statistic.
    """
    result = pandas.DataFrame()

    for subdir in Path(topdir).iterdir():
        if not subdir.is_dir() or '.' in subdir.name:
            continue
        s = sample.Sample(subdir)
        pos = s.positionsdf.copy(deep=True)
        pos['All'] = s.df.sum(axis=1).astype('Int64')
        pos['Capture Edge Distance'] = spatial.capture_edge_distance(pos)
        pos['Tissue Edge Distance'] = spatial.tissue_edge_distance(pos)
        bd = get_border_distance_pvalues(pos, btype=btype)[:9]
        bd['sample'] = subdir.name
        bd.reset_index(names='distance', inplace=True)
        bd = bd.pivot(index='sample', columns='distance')

        result = pandas.concat([result, bd])
    return result


def get_border_distance_pvalues(p, maxdist=4, btype='Capture'):
    """
    parameters
    ----------
        p: a pandas Dataframe from 10X tissue_positions.csv with the following
            additional columns:
                'All': the sum of gene reads at the spots
                'Tissue Edge Distance': from spatial.tissue_edge_distance()
                'Capture Edge Distance': from spatial.capture_edge_distance()
        btype: the type of border
            {'Capture', 'Tissue'}
    returns a dictinary of border distance to interior pvalues
    """
    distance_col = 'Capture Edge Distance'
    if btype == 'Tissue':
        distance_col = 'Tissue Edge Distance'
    elif btype != 'Capture':
        raise ValueError('btype must be Capture or Tissue')

    cols = ['ks_pval', 'ks_stat',
            'tt_pval', 'tt_stat',
            'mannwhitney_pval', 'mannwhitney_stat',
            'border_cells', 'interior_cells',
            'fdr', 'border_mu', 'border_sigma',
            'interior_mu', 'interior_sigma']
    df = pandas.DataFrame(index=range(1, maxdist + 1),
                          columns=cols,
                          dtype='float64[pyarrow]')
    for distance in range(1, maxdist + 1):
        p['border'] = 'Whatever'
        p.loc[p[distance_col] == distance, 'border'] = 'Border'
        p.loc[p[distance_col] > distance, 'border'] = 'Interior'
        num_border_cells = len(p[p.border == 'Border'])
        num_interior_cells = len(p[p.border == 'Interior'])

        df.at[distance, 'border_cells'] = num_border_cells
        df.at[distance, 'interior_cells'] = num_interior_cells

        if num_border_cells < 2 or num_interior_cells < 2:
            continue
        else:
            t_test = ttest_ind(p.loc[p.border == 'Border', 'All'],
                               p.loc[p.border == 'Interior', 'All'],
                               nan_policy='omit',
                               equal_var=False)
            df.at[distance, 'tt_pval'] = round(t_test.pvalue, 4)
            df.at[distance, 'tt_stat'] = round(t_test.statistic, 4)
            ks_test = kstest(p.loc[p.border == 'Border', 'All'],
                             p.loc[p.border == 'Interior', 'All'])
            df.at[distance, 'ks_pval'] = round(ks_test.pvalue, 4)
            df.at[distance, 'ks_stat'] = round(ks_test.statistic, 4)
            mw_test = mannwhitneyu(p.loc[p.border == 'Border', 'All'],
                                   p.loc[p.border == 'Interior', 'All'],
                                   nan_policy='omit')
            df.at[distance, 'mannwhitney_pval'] = round(mw_test.pvalue, 4)
            df.at[distance, 'mannwhitney_stat'] = round(mw_test.statistic, 4)

            results = df.loc[distance,
                             ['ks_pval', 'tt_pval', 'mannwhitney_pval']]

            fdr = false_discovery_control(results).min()
            df.at[distance, 'fdr'] = fdr

            if not isnan(df.at[distance, 'fdr']) and df.at[distance, 'fdr'] < 0.05:
                df.at[distance, 'border_mu'] = p.loc[p.border == 'Border', 'All'].mean()
                df.at[distance, 'border_sigma'] = p.loc[p.border == 'Border', 'All'].std()
                df.at[distance, 'interior_mu'] = p.loc[p.border == 'Interior', 'All'].mean()
                df.at[distance, 'interior_sigma'] = p.loc[p.border == 'Interior', 'All'].std()
            else:
                df.at[distance, 'border_mu'] = pandas.NA
                df.at[distance, 'border_sigma'] = pandas.NA
                df.at[distance, 'interior_mu'] = pandas.NA
                df.at[distance, 'interior_sigma'] = pandas.NA

    return df
