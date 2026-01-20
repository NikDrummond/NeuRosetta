""" Functions for reading and writing .swc files"""

###


### check swc columns

def _check_swc_columns(df, error_type=ValueError):
    """
    Checks if required columns exist in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        required_columns (list): List of required column names.
        error_type (Exception): Type of exception to raise (default: ValueError).
        
    Raises:
        error_type: If any required column is missing.
    """
    required_columns = ['node_id','type','x','y','z','radius','parent_id']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise error_type(f"Missing required columns: {missing}")