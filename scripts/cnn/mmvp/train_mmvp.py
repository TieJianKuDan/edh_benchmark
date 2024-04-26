import sys

sys.path.append("./")

from core.cnn.mmvp.mmvp_pl import MMVPPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=MMVPPL,
        config_dir="scripts/cnn/mmvp/config.yaml",
        log_dir="./logs/",
        log_name="mmvp_2",
        ckp_dir="ckps/mmvp_2"
    )
    cc.train()

if __name__ == "__main__":
    main()
    