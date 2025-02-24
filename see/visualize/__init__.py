from see.visualize.event_low_light_batch import EventLowLightBatchVisualizer


def get_visulization(config):
    return EventLowLightBatchVisualizer(visdir=config.folder, tag=config.tag)
