import base64


def encode_image(data):
    """
    Encodes binary image data to a base64 ASCII string.
    Args:
        data (bytes): The binary image data to encode.
    Returns:
        str: The base64 encoded ASCII string representation of the image data.
    """

    return base64.b64encode(data).decode('ascii')
