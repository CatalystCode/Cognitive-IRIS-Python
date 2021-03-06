# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator 1.0.1.0
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class ImageLabelCreateModel(Model):
    """ImageLabelCreateModel.

    :param tag_id:
    :type tag_id: str
    :param region:
    :type region: :class:`ImageBoundingBox <training.models.ImageBoundingBox>`
    """

    _attribute_map = {
        'tag_id': {'key': 'TagId', 'type': 'str'},
        'region': {'key': 'Region', 'type': 'ImageBoundingBox'},
    }

    def __init__(self, tag_id=None, region=None):
        self.tag_id = tag_id
        self.region = region
