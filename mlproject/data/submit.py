from pandas import DataFrame

def submit(id_test, preds, args):
    """
        Function to create sumbit file for Kaggle competition
    """
    df_submit = DataFrame({'test_id':id_test, 'is_duplicate': preds})
    file_name = "{}/submit_{}_{}_{:.5f}_0.00000.csv.gz".format(*args)
    df_submit.to_csv(file_name, index=False, compression='gzip')
