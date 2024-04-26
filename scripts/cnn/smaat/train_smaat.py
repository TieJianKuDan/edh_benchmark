import sys

sys.path.append("./")

from core.cnn.smaat.unet_pl import SmaAtUNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=SmaAtUNetPL,
        config_dir="scripts/cnn/smaat/config.yaml",
        log_dir="./logs/",
        log_name="smaat_2",
        ckp_dir="ckps/smaat_2"
    )
    cc.train()

if __name__ == "__main__":
    main()
    