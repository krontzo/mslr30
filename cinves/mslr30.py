import csv
import pathlib
import tomllib
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Bbox
import matplotlib.colors as mcolors



class GabyColors:
    """
    YOLO Pose keypoints
    6 to 11
    Left Shoulder
    Right Shoulder
    Left Elbow
    Right Elbow
    Left Wrist
    Right Wrist
    """

    kp_list = [10, 9, 8, 7, 6, 5]  # keypoints de ambas manos, hombros y codos # FIXME: manos, codos y hombros (comentario de Gaby)
    color_names = ['steelblue', 'orchid', 'indigo',
                   'brown', 'seagreen', 'tomato']
    dico = dict(left=dict(shoulder=3, elbow=4, wrist=5),
        right=dict(shoulder=2, elbow=1, wrist=0))
    def __init__(self):
        self.colors = []
        for color in self.color_names:
            self.colors.append(mcolors.to_hex(color))
        """
        ['#4682b4', '#da70d6', '#4b0082',
         '#a52a2a', '#2e8b57', '#ff6347']
        """

    def get_color_for_mslr(self):
        idx = [0, 2, 4, 5, 3, 1]
        cls = [self.colors[i] for i in idx]
        return cls

    def __getitem__(self, key):
        arm, joint = key
        assert isinstance(arm, str)
        assert isinstance(joint, str)
        tmp = self.get_color_for_mslr()
        idx = self.dico[arm][joint]
        return tmp[idx]


class KeypointsSign:
    order = dict(body=5,face=20,left_hand=21)
    nb_keypoints = 67
    left_shoulder = [0, 5, 10]
    left_elbow = [2, 7, 12]
    left_wrist = [75, 96, 117]
    right_shoulder = [1, 6, 11]
    right_elbow = [3, 8, 13]
    right_wrist = [138, 159, 180]

    def __init__(self, raw_data, identifier=None):
        row = raw_data[0]
        assert len(row) == self.nb_keypoints * 3
        self.data = np.asarray(raw_data)
        self.identifier = identifier
        self.nb_frames, _ = self.data.shape
        shape = (self.nb_frames, 3, -1)
        cnt = np.array(
            [0,
             self.order["body"],
             self.order["face"],
             self.order["left_hand"],
            ]
        )
        i, off = 0, cnt.cumsum() * 3
        self.body = self.data[:, off[i]:off[i+1]].reshape(shape)
        i += 1
        self.face = self.data[:, off[i]:off[i+1]].reshape(shape)
        i += 1
        self.left_hand = self.data[:, off[i]:off[i+1]].reshape(shape)
        i += 1
        self.right_hand = self.data[:, off[i]:].reshape(shape)

    def __str__(self):
        msg = f"Keypoints({self.identifier}, {self.data.shape})"
        return msg

    def direct_arm_keypoints(self):
        lw_x, lw_y, lw_z = self.left_wrist
        le_x, le_y, le_z = self.left_elbow
        ls_x, ls_y, ls_z= self.left_shoulder
        rw_x, rw_y, rw_z = self.right_wrist
        re_x, re_y, re_z = self.right_elbow
        rs_x, rs_y, rs_z = self.right_shoulder
        dat = self.data
        lst = np.array([
            dat[:, [rw_x, rw_y, rw_z]],
            dat[:, [re_x, re_y, re_z]],
            dat[:, [rs_x, rs_y, rs_z]],
            dat[:, [ls_x, ls_y, ls_z]],
            dat[:, [le_x, le_y, le_z]],
            dat[:, [lw_x, lw_y, lw_z]],
        ])

        kpts = dict(
                left=dict(shoulder=lst[3], elbow=lst[4], wrist=lst[5]),
                right=dict(shoulder=lst[2], elbow=lst[1], wrist=lst[0])
        )
        return lst


    def arm_keypoints(self):
        lst = np.array([
                self.right_hand[:, :, 0], self.body[:, :, 3], self.body[:, :, 1],
                self.body[:, :, 0], self.body[:, :, 2], self.left_hand[:, :, 0]
              ])
        kpts = dict(
                left=dict(shoulder=lst[3], elbow=lst[4], wrist=lst[5]),
                right=dict(shoulder=lst[2], elbow=lst[1], wrist=lst[0])
        )
        print(lst.shape, self.identifier)
        lst_direct = self.direct_arm_keypoints()
        assert np.all(lst == lst_direct)
        return kpts, lst

    def metadata(self):
        mdata = {'Creator': 'lsm74',
                 'Title': f"'{self.identifier}'",
                 'Source': 'mslr',
        }

    def plot_body(self):
        frame = 9
        fig, axs = plt.subplots(1, 4)
        kpts = self.body[9]
        x, y, z = kpts
        x_f, y_f, z_f = face = self.face[9]
        x_lh, y_lh, z_lh = l_hand = self.left_hand[9]
        x_rh, y_rh, z_rh = r_hand = self.right_hand[9]
        print(l_hand)
        axs[0].axis("equal")
        axs[0].plot(x, y, '.')
        #axs[1].axis("equal")
        #axs[1].plot(x_f, y_f, '.')
        axs[2].axis("equal")
        axs[2].plot(x, -y, '.')
        axs[2].plot(x_f, -y_f, '.')
        axs[2].plot(x_lh, -y_lh, '.')
        axs[2].plot(x_rh, -y_rh, '.')
        dico, kpts = self.arm_keypoints()
        xk, yk, _ = kpts[:, frame, :].T
        axs[0].plot(xk, yk, 'rx:')

        for idx, kpt in enumerate(kpts):
            print(idx, kpt.shape)
            axs[1].plot(kpt[:, 0], kpt[:, 1], "-", label=f"{idx}")

        for ida, (arm, val) in enumerate(dico.items()):
            print(idx, kpt.shape)
            for idx, (label, kpt) in enumerate(val.items()):
                print(idx, kpt.shape)
                axs[-1].plot(kpt[:, 0], kpt[:, 1], "-", label=f"{idx}: {arm} {label}")
        for i in [1, -1]:
            axs[i].axis("equal")
            axs[i].set_title("Arm Keypoints")
            axs[i].yaxis.set_inverted(True)
            axs[i].legend()
        fig.savefig("/tmp/shape.svg", format="svg", metadata=self.metadata())
        plt.suptitle(self.identifier)

        from io import StringIO
        f = StringIO()
        print(f.getvalue())

        self.plot_shape()
        plt.show()

    def plot_shape(self, show=False, saveto=None):
        dico, kpts = self.arm_keypoints()
        box = Bbox.null()  # FIXME: delete if not needed
        colors = GabyColors()
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 6.4))
        for ida, (arm, val) in enumerate(dico.items()):
            for idx, (label, kpt) in enumerate(val.items()):
                gid = f"{arm} {label}"
                axs.plot(kpt[:, 0], kpt[:, 1], "-", label=gid, gid=gid, color=colors[arm, label])
        axs.axis("equal")
        axs.yaxis.set_inverted(True)
        axs.set_axis_off()
        # FIXME: not completed, # self.shape_to_svg()
        if saveto:
            path = saveto / (self.identifier + ".png")
            fig.savefig(path,
                        #bbox_inches="tight",
                        format="png")
            # img = plt.imread(str(path))
            # print(img.shape)
        if show:
            fig.suptitle(self.identifier)
            plt.show()
        else:
            plt.close()

    def shape_to_svg(self, width=640):
        """
        WIP: custom svg output using polyline
        """
        height = width
        colors = ["rgb(100, 149, 237)", "#6495ED"]
        header = f'<svg height="{height}" width="width" xmlns="http://www.w3.org/2000/svg">'
        poly = '<polyline points="{}" style="{}" />'
        style = 'fill:none;stroke:{};stroke-width:{}'
        footer = "</svg>"
        pts = "10,10 20,30 100,150 200,200 320,320 600,320 600,600"
        print(header,
              poly.format(pts, style.format('red', 3)),
              footer, sep="\n")


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


_DEFAULT_CFG_DIR = "."

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
        #sign_a.plot_shape(show=show, saveto=sample_path)
        sign_a.plot_shape(show=True, saveto=None)


def main():
    lst = get_config_files()
    for cfg in lst:
        print(cfg)
        dico = get_config_from(cfg)
        dataset = Dataset(**dico["mslr"])
        signs = dataset.signs
        print(signs)
        for sign in signs[:]:
            samples = dataset.get_file_list(sign)
            samples_path = pathlib.Path('~/Public/excluded').expanduser()
            for sample in samples:
                convert_sample_to_shape(sample, samples_path)
                break
 

if __name__ == "__main__":
    main()
