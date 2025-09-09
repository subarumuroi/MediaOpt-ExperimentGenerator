import pandas as pd

def load_and_filter_yarrowia_matrix(path):
    '''
    loads g/L data where rows are compounds and columns are samples, requires there to be
    pubchem in column[1] and annotation (i.e., vitamins, elements, glucose) in last column.
    
    Args:
        path (str): path to csv file
        
    Returns:
        yar_df_element (pd.DataFrame): filtered dataframe for elements
        yar_df_vitamin (pd.DataFrame): filtered dataframe for vitamins
    '''
    # read in g/L data for dimensional reduction 
    df = pd.read_csv(path)

    # filter rows that are considered not important
    compounds_to_remove = [
        'Dipotassium phosphate', 'Disodium edta dihydrate', 'EDTA', 'Monopotassium phosphate',
        'Potassium hydroxide', 'Sodium citrate dihydrate', 'Sodium phosphate',
        'Uracil', 
        # also remove redundant elements 
        'Calcium sulphate dihydrate', 'Iron (III) chloride tetrahydrate', 'Manganese (II) sulphate monohydrate',
        'Nickel sulphate heptahydrate', 'Sodium chloride', 'Zinc chloride'

    ]

    df = df[~df['Compound'].isin(compounds_to_remove)].copy()

    # remove pubchemid
    df.drop(df.columns[1], axis=1, inplace=True)

    # separate numeric columns for transformation
    num_df = df.select_dtypes(include='number')
    labels = df[['Compound', 'Annotate']]

    yar_df = df.filter(like='_YAR')
    #cer_df = df_log.filter(like='_CER')
    #pic_df = df_log.filter(like='_PIC')

    yar_df_labeled = pd.concat([labels, yar_df], axis=1)

    # Define which columns are sample measurements (everything except metadata)
    sample_cols = yar_df_labeled.columns.difference(['Compound', 'Annotate'])

    # Filter to keep rows where at least one sample value ≠ 0
    yar_df_labeled = yar_df_labeled[
        yar_df_labeled['Annotate'].isin(['Vitamin', 'Element']) &
        ~(yar_df_labeled[sample_cols] == 0).all(axis=1)
    ].copy()

    yar_df_element = yar_df_labeled[yar_df_labeled['Annotate'] == 'Element'].copy()
    yar_df_vitamin = yar_df_labeled[yar_df_labeled['Annotate'] == 'Vitamin'].copy()

    return yar_df_element, yar_df_vitamin