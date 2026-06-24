import csv
import pathlib
import tomllib
from dataclasses import dataclass, field

from visualize_sign import KeypointsSign

@dataclass
class Dataset:
    name: str
    number_of_signs: int
    base_path: str
    samples_per_sign: int
    total_samples: int
    filename_pattern: str  # {sign}, {id}
    signs: list = field(default_factory=list)
    helper_cmd: str = ""
    comments: str = ""

    def get_file_list(self, sign):
        path = pathlib.Path(self.base_path).expanduser()
        pattern = self.filename_pattern.format(
            sign=sign, id="*"
        )
        lst = list(path.rglob(pattern))
        return lst


_DEFAULT_CFG_DIR = "~/.config/lsm"

def get_config_files():
    lst = pathlib.Path(_DEFAULT_CFG_DIR).expanduser()
    lst = list(lst.glob("data*.toml"))
    return lst


def get_config_from(cfg):
    with cfg.open('rb') as cfgfile:
        data = tomllib.load(cfgfile)
    return data


def get_data_from_csv(path):
    with path.open() as csvfile:
        data = csv.reader(csvfile)
        data = [_ for _ in data]
    return data

def preprocess_mslr_data(data):
    """
    First row and first column are headers
    """
    data = [[float(val) for val in row[1:]] for row in data[1:]]
    return data

def convert_sample_to_shape(sample_filename, sample_path, show=False):
        # Read one file
        data_sample_a = get_data_from_csv(sample_filename)
        data_sample_a = preprocess_mslr_data(data_sample_a)
        assert len(data_sample_a) == 20
        assert len(data_sample_a[0]) == 67 * 3
        sign_a = KeypointsSign(data_sample_a, sample_filename.stem)
        sign_a.plot_shape(show=show, saveto=sample_path)


def main():
    lst = get_config_files()
    for cfg in lst:
        print(cfg)
        dico = get_config_from(cfg)
        dataset = Dataset(**dico["mslr"])
        signs = dataset.signs
        print(signs)
        for sign in signs:
            samples = dataset.get_file_list(sign)
            samples_path = pathlib.Path('~/Public/excluded').expanduser()
            for sample in samples:
                convert_sample_to_shape(sample, samples_path)

if __name__ == "__main__":
    main()
