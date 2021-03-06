# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator 1.0.1.0
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class ImageUploadSummary(Model):
    """ImageUploadSummary.

    :param is_successful:
    :type is_successful: bool
    :param images:
    :type images: list of :class:`ImageUploadResultModel
     <training.models.ImageUploadResultModel>`
    """

    _attribute_map = {
        'is_successful': {'key': 'IsSuccessful', 'type': 'bool'},
        'images': {'key': 'Images', 'type': '[ImageUploadResultModel]'},
    }

    def __init__(self, is_successful=None, images=None):
        self.is_successful = is_successful
        self.images = images

    def get_isSuccessful(self):
        return self.is_successful