import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Bbox

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

class KeypointsSign:
    order = dict(body=5,face=20,left_hand=21)
    nb_keypoints = 67

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

    def arm_keypoints(self):
        lst = np.array([
                self.right_hand[:, :, 0], self.body[:, :, 3], self.body[:, :, 1],
                self.body[:, :, 0], self.body[:, :, 2], self.left_hand[:, :, 0]
              ])
        kpts = dict(
                left=dict(shoulder=lst[3], elbow=lst[4], wrist=lst[5]),
                right=dict(shoulder=lst[2], elbow=lst[1], wrist=lst[0])
        )
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
        box = Bbox.null()
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 6.4))
        for ida, (arm, val) in enumerate(dico.items()):
            for idx, (label, kpt) in enumerate(val.items()):
                gid = f"{arm} {label}"
                axs.plot(kpt[:, 0], kpt[:, 1], "-", label=gid, gid=gid)
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
