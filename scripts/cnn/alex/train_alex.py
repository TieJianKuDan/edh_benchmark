import sys

sys.path.append("./")

from core.cnn.alex.alex_pl import AlexNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import DistriPLDM


def main():
    cc = ClassicConf(
        dm_class=DistriPLDM,
        model_class=AlexNetPL,
        config_dir="scripts/cnn/alex/config.yaml",
        log_dir="./logs/",
        log_name="alex",
        ckp_dir="ckps/alex"
    )
    cc.train()

if __name__ == "__main__":
    main()
    