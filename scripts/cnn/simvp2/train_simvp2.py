import sys

sys.path.append("./")

from core.cnn.simvp2.simvp2_pl import SimVP2PL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=SimVP2PL,
        config_dir="scripts/cnn/simvp2/config.yaml",
        log_dir="./logs/",
        log_name="simvp2_3",
        ckp_dir="ckps/simvp2_3"
    )
    cc.train()

if __name__ == "__main__":
    main()
    