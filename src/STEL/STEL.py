from STEL.utility.set_for_global import CORRECT_ALTERNATIVE_COL, ALTERNATIVE12_COL, ANCHOR2_COL, ALTERNATIVE11_COL
from STEL.utility.eval_on_tasks import eval_model


def get_STEL_Or_Content_from_STEL(pd_stel_instances):
    """
        create the STEL-Or-Content task from oroginal STEl instances

        :param pd_stel_instances: pandas dataframe of original STEL instances
    """
    for row_id, row in pd_stel_instances.iterrows():
        if row[CORRECT_ALTERNATIVE_COL] == 1:
            # S1-S2 is correct order, i.e., style of A1 and S1 is the same
            pd_stel_instances.at[row_id, ALTERNATIVE12_COL] = row[ANCHOR2_COL]
        else:
            # S2-S1 is correct order, i.e., style of A1 and S2 is the same
            pd_stel_instances.at[row_id, ALTERNATIVE11_COL] = row[ANCHOR2_COL]
    return pd_stel_instances


def eval_on_STEL(style_objects):
    print("Performance on original STEL tasks")
    org_STEL_result = eval_model(style_objects=style_objects)
    pd_stel_instances = org_STEL_result["stel_tasks"]
    pd_stel_instances = get_STEL_Or_Content_from_STEL(pd_stel_instances)
    print("Performance on STEL-Or-Content tasks")
    STEL_or_content_result = eval_model(style_objects=style_objects, stel_instances=pd_stel_instances, eval_on_triple=True)
    return org_STEL_result, STEL_or_content_result
