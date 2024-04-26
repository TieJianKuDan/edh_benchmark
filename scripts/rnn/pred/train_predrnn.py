import sys

sys.path.append("./")

from core.rnn.pred.predrnn_pl import PredRNNPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=PredRNNPL,
        config_dir="scripts/rnn/pred/config.yaml",
        log_dir="./logs/",
        log_name="predrnn_2",
        ckp_dir="ckps/predrnn_2"
    )
    cc.train()

if __name__ == "__main__":
    main()
    