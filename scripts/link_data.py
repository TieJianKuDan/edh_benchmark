import os

def ln_train_data():
    for i in range(2020, 2023):
        os.makedirs(f"data/train/edh/{i}", exist_ok=True)
        os.system(
            f"ln -s /root/autodl-fs/edh/{i}/* data/train/edh/{i}"
        )
        os.makedirs(f"data/train/era5/{i}", exist_ok=True)
        os.system(
            f"ln -s /root/autodl-fs/era5/{i}/* data/train/era5/{i}"
        )

def ln_test_data():
    for i in range(2023, 2024):
        os.makedirs(f"data/test/edh/{i}", exist_ok=True)
        os.system(
            f"ln -s /root/autodl-fs/edh/{i}/* data/test/edh/{i}"
        )
        os.makedirs(f"data/test/era5/{i}", exist_ok=True)
        os.system(
            f"ln -s /root/autodl-fs/era5/{i}/* data/test/era5/{i}"
        )

def ln_temp_data():
    for i in range(2023, 2024):
        os.makedirs(f"data/temp/edh/{i}", exist_ok=True)
        os.system(
            f"ln -s /root/autodl-fs/edh/{i}/* data/temp/edh/{i}"
        )
        os.makedirs(f"data/temp/era5/{i}", exist_ok=True)
        os.system(
            f"ln -s /root/autodl-fs/era5/{i}/* data/temp/era5/{i}"
        )

if __name__=="__main__":
    ln_train_data()
    ln_test_data()