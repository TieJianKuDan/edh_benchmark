import sys

sys.path.append("./")

from core.rnn.conv.convlstm_pl import ConvLSTMPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=ConvLSTMPL,
        config_dir="scripts/rnn/conv/config.yaml",
        log_dir="./logs/",
        log_name="convlstm_2",
        ckp_dir="ckps/convlstm_2"
    )
    cc.train()

if __name__ == "__main__":
    main()
    