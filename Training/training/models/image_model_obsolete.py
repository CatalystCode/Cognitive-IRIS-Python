# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator 1.0.1.0
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class ImageModelObsolete(Model):
    """Image model to be sent as JSON.

    :param id:
    :type id: str
    :param width:
    :type width: int
    :param height:
    :type height: int
    :param image_hash:
    :type image_hash: long
    :param creation_time_stamp:
    :type creation_time_stamp: datetime
    :param mime_type:
    :type mime_type: str
    :param bounding_boxes:
    :type bounding_boxes: list of :class:`ImageRegion
     <training.models.ImageRegion>`
    :param performance:
    :type performance: :class:`ImageResultModel
     <training.models.ImageResultModel>`
    """

    _attribute_map = {
        'id': {'key': 'Id', 'type': 'str'},
        'width': {'key': 'Width', 'type': 'int'},
        'height': {'key': 'Height', 'type': 'int'},
        'image_hash': {'key': 'ImageHash', 'type': 'long'},
        'creation_time_stamp': {'key': 'CreationTimeStamp', 'type': 'iso-8601'},
        'mime_type': {'key': 'MimeType', 'type': 'str'},
        'bounding_boxes': {'key': 'BoundingBoxes', 'type': '[ImageRegion]'},
        'performance': {'key': 'Performance', 'type': 'ImageResultModel'},
    }

    def __init__(self, id=None, width=None, height=None, image_hash=None, creation_time_stamp=None, mime_type=None, bounding_boxes=None, performance=None):
        self.id = id
        self.width = width
        self.height = height
        self.image_hash = image_hash
        self.creation_time_stamp = creation_time_stamp
        self.mime_type = mime_type
        self.bounding_boxes = bounding_boxes
        self.performance = performance