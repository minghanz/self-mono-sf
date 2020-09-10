"""This file is to test the c3d loss behavior using a pickled input"""
from c3d import C3DLoss

if __name__ == "__main__":
    c3d_loss = C3DLoss(flow_mode=True)
    cfg_file = "/home/minghanz/self-mono-sf/scripts/c3d_config.txt"
    pkl_file = "/home/minghanz/self-mono-sf/debug_c3d/nan_dicts.pkl"
    c3d_loss.parse_opts(f_input=cfg_file)
    c3d_loss.debug_flow_inspect_input(pkl_file)