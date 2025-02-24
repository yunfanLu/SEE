import pudb
from absl.logging import info
from absl.testing import absltest
from torch.utils.data import DataLoader
from tqdm import tqdm

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.datasets.sde import get_seeing_dynamic_with_event_dataset_all, get_seeing_dynamic_with_event_dataset_forset
from see.utils import print_batch
from see.utils.event_representation_builder import VoxelGridConfig
from see.visualize.event_low_light_batch import EventLowLightBatchVisualizer


class SeeingDynamicWithEventVideoDatasetTest(absltest.TestCase):
    def setUp(self):
        crop_h, crop_w = 256, 256
        root = "dataset/CVPR24/0-Low-Light-CVPR24/"
        event_representation_config = VoxelGridConfig(channel=80, H=260, W=346, to_bin=True, scale=1.0)
        self.train, self.test = get_seeing_dynamic_with_event_dataset_all(
            root, 1, crop_h, crop_w, event_representation_config
        )
        self.visualizer = EventLowLightBatchVisualizer("testdata", "test", vis_intermediate=False)

    def test_foreach(self):
        info(f"train: {len(self.train)}")
        info(f"test: {len(self.test)}")
        for i, batch in enumerate(self.train):
            print_batch(batch, ppfun=info)
            return

    def test_dataloader(self):
        train = DataLoader(
            dataset=self.train,
            batch_size=20,
            shuffle=False,
            num_workers=36,
            pin_memory=True,
            drop_last=True,
        )
        for i, batch in tqdm(enumerate(train)):
            print_batch(batch, ppfun=info)
            self.visualizer(batch)
            break

        for i, batch in tqdm(enumerate(train)):
            print(f"video name: {batch[ELB.VIDEO_NAME]}, {batch[ELB.FRAME_NAME]}", flush=True)

        test = DataLoader(
            dataset=self.test,
            batch_size=10,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        for i, batch in tqdm(enumerate(test)):
            print(f"video name: {batch[ELB.VIDEO_NAME]}, {batch[ELB.FRAME_NAME]}")


if __name__ == "__main__":
    absltest.main()
