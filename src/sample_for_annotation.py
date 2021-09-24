"""
    sample questions from total for later annotation upload
"""
import logging
import set_for_global
import pandas as pd
import os

from set_for_global import ID_COL, IN_SUBSAMPLE_COL, VAL_SIMPLICITY, VAL_FORMALITY, STYLE_DIMS, FORMAL_KEY, SIMPLE_KEY

# RATIO_COMPLEX_TO_FORMAL = 0.68/0.88
COMPLEX_ACC = 0.68
FORMAL_ACC = 0.88
SAMPLE_SIZE = 504


def main(tsv_q, tsv_samples=None, sample_size=SAMPLE_SIZE, complex_acc=COMPLEX_ACC, formal_acc=0.88,
         style_dims=STYLE_DIMS, add_formal_0=180, add_simple_0=324, nbr_samples=1):
    set_for_global.set_global_seed(w_torch=False)
    set_for_global.set_logging()

    logging.info("Sampling {} questions from dimensions {} with complex to formal ratio {}"
                 .format(sample_size, style_dims, complex_acc/formal_acc))

    assert(len(tsv_samples) >= 2, 'no samples given ...')
    assert(nbr_samples >= 1, 'at least one sample has to be generated')

    quad_df = pd.read_csv(tsv_q, sep='\t')
    subsamples_df = pd.read_csv(tsv_samples[0], sep='\t', index_col=0)
    for tsv_sample in tsv_samples[1:]:
        cur_subsample = pd.read_csv(tsv_sample, sep='\t', index_col=0)
        subsamples_df = pd.concat([subsamples_df, cur_subsample])
        # subsamples_df = subsamples_df.merge(cur_subsample, how='outer')
        assert(len(subsamples_df[subsamples_df[IN_SUBSAMPLE_COL] == False]), 0)
    sample_ids = list(subsamples_df[ID_COL])

    for i in range(nbr_samples):
        sampleable_quads = quad_df[~quad_df[ID_COL].isin(sample_ids)]
        nbr_simple_samples = len(subsamples_df[(subsamples_df[ID_COL].str.contains(SIMPLE_KEY))])
        nbr_formal_samples = len(subsamples_df[(subsamples_df[ID_COL].str.contains(FORMAL_KEY))])
        if i == 0:
            add_formal = add_formal_0
            add_simple = add_simple_0
        else:
            add_simple = int(sample_size*formal_acc/(complex_acc + formal_acc))
            add_formal = sample_size - add_simple

        new_sampled_quads = sample_qs(sampleable_quads, add_formal=add_formal, add_simple=add_simple)
        subsamples_df = pd.concat([subsamples_df, new_sampled_quads])
        sample_ids = list(subsamples_df[ID_COL])

        subsample_file_folder = os.path.dirname(tsv_q)
        subsample_file_name = "fullsample-{}_quad_questions".format(i+1)
        subsample_file_name += "_{}-{}".format(VAL_SIMPLICITY, int(add_simple))
        subsample_file_name += "_{}-{}".format(VAL_FORMALITY, int(add_formal))
        subsample_file_ending = ".tsv"
        new_sampled_quads.to_csv(subsample_file_folder + '/' + subsample_file_name + subsample_file_ending, sep='\t',
                                 index=False)

        logging.info('Estimated new number of formal STLE qs {}'.format((add_formal + nbr_formal_samples)*formal_acc))
        logging.info('Estimated new number of simple STLE qs {}'.format((add_simple + nbr_simple_samples)*complex_acc))

    return subsamples_df, quad_df[~quad_df[ID_COL].isin(sample_ids)], quad_df


def sample_qs(sampleable_quads, add_formal, add_simple):
    for i, val_type in enumerate(STYLE_DIMS):
        if val_type == VAL_SIMPLICITY:
            cur_sample_indices = sampleable_quads[sampleable_quads[ID_COL].str.contains(SIMPLE_KEY)] \
                .sample(int(add_simple), replace=False).index
        else:
            cur_sample_indices = sampleable_quads[sampleable_quads[ID_COL].str.contains(FORMAL_KEY)] \
                .sample(int(add_formal), replace=False).index
        if i == 0:
            sample_indices = cur_sample_indices
        else:
            sample_indices = sample_indices.union(cur_sample_indices)
    sampleable_quads.loc[sample_indices, IN_SUBSAMPLE_COL] = True
    sampleable_quads = sampleable_quads.loc[sample_indices]
    return sampleable_quads


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()
