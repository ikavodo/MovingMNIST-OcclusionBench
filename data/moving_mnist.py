import torch
from torch.utils.data import Dataset
import random

IMG_SIZE = 28


class MovingMNIST(Dataset):
    def __init__(
            self,
            base_mnist,
            T: int = 20,
            canvas: int = 64,
            vmin: int = -3,
            vmax: int = 3,
            seed: int = 0,
    ):
        assert canvas >= IMG_SIZE
        self.base = base_mnist
        self.T = T
        self.canvas = canvas
        self.vmin = vmin
        self.vmax = vmax
        self.seed = seed

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _bounce(pos, v, lo, hi):
        nxt = pos + v
        if nxt < lo:
            return lo, abs(v)
        if nxt > hi:
            return hi, -abs(v)
        return nxt, v

    @staticmethod
    def _zero_wrap(frame, dy, dx):
        if dy > 0:
            frame[:, :dy, :] = 0
        elif dy < 0:
            frame[:, dy:, :] = 0
        if dx > 0:
            frame[:, :, :dx] = 0
        elif dx < 0:
            frame[:, :, dx:] = 0
        return frame

    def _sample_vxy(self, g: torch.Generator):
        while True:
            vx = torch.randint(self.vmin, self.vmax + 1, (1,), generator=g).item()
            vy = torch.randint(self.vmin, self.vmax + 1, (1,), generator=g).item()
            if vx != 0 or vy != 0:
                return vx, vy

    def __getitem__(self, idx):
        img, label = self.base[idx]
        H = W = self.canvas
        xmax = W - IMG_SIZE
        ymax = H - IMG_SIZE

        g = torch.Generator()
        g.manual_seed(self.seed + idx)

        x = torch.randint(0, xmax + 1, (1,), generator=g).item()
        y = torch.randint(0, ymax + 1, (1,), generator=g).item()
        vx, vy = self._sample_vxy(g)

        frame = torch.zeros(1, H, W, dtype=img.dtype)
        frame[:, y:y + IMG_SIZE, x:x + IMG_SIZE] = img

        video = [frame.clone()]
        shifts = []
        for _ in range(self.T - 1):
            nx, vx = self._bounce(x, vx, 0, xmax)
            ny, vy = self._bounce(y, vy, 0, ymax)
            dx, dy = nx - x, ny - y
            shifts.append((dy, dx))
            frame = torch.roll(frame, shifts=(dy, dx), dims=(1, 2))
            frame = self._zero_wrap(frame, dy, dx)
            x, y = nx, ny
            video.append(frame.clone())

        meta = {"base_idx": int(idx), "seed": int(self.seed + idx)}
        return torch.stack(video, dim=0), int(label), torch.tensor(shifts, dtype=torch.long), meta


# wrapper which extracts k frames
class MovingMNISTFrames(Dataset):
    def __init__(self, moving_ds: Dataset, frame_seed: int = 0):
        self.ds = moving_ds
        self.frame_seed = frame_seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        video, y, _, meta = self.ds[idx]
        rng = random.Random(self.frame_seed + idx)
        t = rng.randrange(video.shape[0])
        x = video[t]
        out_meta = dict(meta)
        out_meta["frame_index"] = int(t)
        return x, y, out_meta
