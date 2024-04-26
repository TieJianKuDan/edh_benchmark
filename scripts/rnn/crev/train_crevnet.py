import sys

sys.path.append("./")

from core.rnn.crev.crev_pl import CrevNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=CrevNetPL,
        config_dir="scripts/rnn/crev/config.yaml",
        log_dir="./logs/",
        log_name="crevnet_3",
        ckp_dir="ckps/crevnet_3"
    )
    cc.train()

if __name__ == "__main__":
    main()
    