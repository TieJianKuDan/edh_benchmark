import sys

sys.path.append("./")

from core.cnn.simvp2.tau_pl import TAUPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=TAUPL,
        config_dir="scripts/cnn/tau/config.yaml",
        log_dir="./logs/",
        log_name="tau_3",
        ckp_dir="ckps/tau_3"
    )
    cc.train()

if __name__ == "__main__":
    main()
    