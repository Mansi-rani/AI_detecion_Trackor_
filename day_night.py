import cv2
import numpy as np


def detect_day_night(image):
    """
    Detect whether an image is Day or Night
    using average image brightness.

    Returns:
        environment: "Day" or "Night"
        brightness: average brightness value
    """

    if image is None:
        return "Day", 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))

    # Brightness threshold
    if brightness < 80:
        environment = "Night"
    else:
        environment = "Day"

    return environment, brightness


def enhance_night_image(image):
    """
    Enhance dark/night images using CLAHE.
    """

    if image is None:
        return image

    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE improves local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced
