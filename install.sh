pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -e ".[goggle]"
