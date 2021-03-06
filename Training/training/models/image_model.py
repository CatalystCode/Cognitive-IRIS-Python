# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator 1.0.1.0
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class ImageModel(Model):
    """Image model to be sent as JSON.

    :param id:
    :type id: str
    :param created_at:
    :type created_at: datetime
    :param hash:
    :type hash: long
    :param width:
    :type width: int
    :param height:
    :type height: int
    :param performance_summary:
    :type performance_summary: str
    :param labels:
    :type labels: list of :class:`ImageLabelModel
     <training.models.ImageLabelModel>`
    """

    _attribute_map = {
        'id': {'key': 'Id', 'type': 'str'},
        'created_at': {'key': 'CreatedAt', 'type': 'iso-8601'},
        'hash': {'key': 'Hash', 'type': 'long'},
        'width': {'key': 'Width', 'type': 'int'},
        'height': {'key': 'Height', 'type': 'int'},
        'performance_summary': {'key': 'PerformanceSummary', 'type': 'str'},
        'labels': {'key': 'Labels', 'type': '[ImageLabelModel]'},
    }

    def __init__(self, id=None, created_at=None, hash=None, width=None, height=None, performance_summary=None, labels=None):
        self.id = id
        self.created_at = created_at
        self.hash = hash
        self.width = width
        self.height = height
        self.performance_summary = performance_summary
        self.labels = labels
