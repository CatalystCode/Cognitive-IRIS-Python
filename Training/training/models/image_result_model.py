# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator 1.0.1.0
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class ImageResultModel(Model):
    """ImageResultModel.

    :param rating: Possible values include: 'None', 'Green', 'Orange', 'Red'
    :type rating: str or :class:`enum <training.models.enum>`
    :param classifications:
    :type classifications: str
    """

    _attribute_map = {
        'rating': {'key': 'Rating', 'type': 'str'},
        'classifications': {'key': 'Classifications', 'type': 'str'},
    }

    def __init__(self, rating=None, classifications=None):
        self.rating = rating
        self.classifications = classifications
