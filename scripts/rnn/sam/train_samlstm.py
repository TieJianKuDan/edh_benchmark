import sys

sys.path.append("./")

from core.rnn.sam.samlstm_pl import SAMConvLSTMPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=SAMConvLSTMPL,
        config_dir="scripts/rnn/sam/config.yaml",
        log_dir="./logs/",
        log_name="samlstm_2",
        ckp_dir="ckps/samlstm_2"
    )
    cc.train()

if __name__ == "__main__":
    main()
    