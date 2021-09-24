import sys, os

sys.path.append(os.path.join('..'))
from to_add_const import LOCAL_TOTAL_DIM_QUAD, LOCAL_STEL_DIM_QUAD
import pandas as pd
from set_for_global import STYLE_TYPE_COL, VAL_FORMALITY, ID_COL, ALTERNATIVE12_COL, ALTERNATIVE11_COL, ANCHOR2_COL, \
    ANCHOR1_COL, VAL_SIMPLICITY


def update_i_f_dicts(sent_id_str, cur_row, firsts_col, seconds_col, f_dict, i_dict):
    sent_id = sent_id_str.split('-')[1]
    if int(sent_id) not in f_dict:
        if sent_id_str.split('-')[0] == 'f':
            f_dict.update({int(sent_id): cur_row[firsts_col]})
            i_dict.update({int(sent_id): cur_row[seconds_col]})
        else:
            i_dict.update({int(sent_id): cur_row[firsts_col]})
            f_dict.update({int(sent_id): cur_row[seconds_col]})
    return f_dict, i_dict


def save_gyafc_ids():
    sent_ids = set()
    inf_sent = {}
    f_sent = {}

    for row_id, row in df_f_exs.iterrows():
        cur_id = row[ID_COL].split('_')
        first_pair = cur_id[1]
        first_id = first_pair.split('-')[1]
        sent_ids.add(int(first_id))
        f_sent, inf_sent = update_i_f_dicts(first_pair, row, ANCHOR1_COL, ANCHOR2_COL, f_sent, inf_sent)
        # print(f_sent, inf_sent)
        second_pair = cur_id[2].split('--')[0]
        second_id = second_pair.split('-')[1]
        sent_ids.add(int(second_id))
        f_sent, inf_sent = update_i_f_dicts(second_pair, row, ALTERNATIVE11_COL, ALTERNATIVE12_COL, f_sent, inf_sent)
        # print(first_pair, second_pair)
    list_sent_ids = list(sent_ids)
    list_sent_ids.sort()
    print(len(list_sent_ids), list_sent_ids)
    print(len(f_sent))
    print(len(inf_sent))
    for sent_id in list_sent_ids:
        print(sent_id, f_sent[sent_id], inf_sent[sent_id])
    with open("formal", "w") as f:
        f.writelines("\n".join([f_sent[sent_id] for sent_id in list_sent_ids]))
    with open("informal.ref0", "w") as f:
        f.writelines("\n".join([inf_sent[sent_id] for sent_id in list_sent_ids]))
    with open("GYAFC_pair_ids", "w") as f:
        f.writelines([str(cur_id) + "\n" for cur_id in list_sent_ids])


def sample_dim(stel_instances, sampling_dim=VAL_FORMALITY, n=100):
    from set_for_global import set_global_seed
    set_global_seed()
    df_sample = stel_instances[stel_instances[STYLE_TYPE_COL] == sampling_dim].sample(n=n)
    index_dropped = df_sample.drop('Unnamed: 0', axis=1)
    # index_dropped = index_dropped.drop('Unnamed: 0.1', axis=1)
    return index_dropped


# stel_dims = pd.read_csv('../' + LOCAL_TOTAL_DIM_QUAD, sep='\t')
stel_dims = pd.read_csv(LOCAL_STEL_DIM_QUAD[0], sep='\t')  # assumes a reference to the 815-815 STEL set
df_f_exs = sample_dim(stel_dims)

save_gyafc_ids()
