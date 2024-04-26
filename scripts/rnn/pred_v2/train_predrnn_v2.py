import sys

sys.path.append("./")

from core.rnn.pred.predrnn_pl import PredRNNV2PL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=PredRNNV2PL,
        config_dir="scripts/rnn/pred_v2/config.yaml",
        log_dir="./logs/",
        log_name="predrnn_v2_1",
        ckp_dir="ckps/predrnn_v2_1"
    )
    cc.train()

if __name__ == "__main__":
    main()
    