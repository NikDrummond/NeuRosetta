from ..core import _Forest

from ..io_utils.swc_utils import export_swc as _write_swc_func
from ..io_utils.nr_utils import save as _save

class Forest(_Forest):

    def __init__(self, trees):
        super().__init__(trees = trees)

    ### io utils
    export_swc = _write_swc_func
    save = _save