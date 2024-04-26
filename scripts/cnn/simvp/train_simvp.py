import sys

sys.path.append("./")

from core.cnn.simvp.simvp_pl import SimVPPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=SimVPPL,
        config_dir="scripts/cnn/simvp/config.yaml",
        log_dir="./logs/",
        log_name="simvp_1",
        ckp_dir="ckps/simvp_1"
    )
    cc.train()

if __name__ == "__main__":
    main()
    