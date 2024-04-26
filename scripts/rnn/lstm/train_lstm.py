import sys

sys.path.append("./")

from core.rnn.lstm.lstm_pl import NStepLSTMPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=NStepLSTMPL,
        config_dir="scripts/rnn/lstm/config.yaml",
        log_dir="./logs/",
        log_name="lstm_3",
        ckp_dir="ckps/lstm_3"
    )
    cc.train()

if __name__ == "__main__":
    main()
    