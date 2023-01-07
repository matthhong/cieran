from coloraide import Color

def color_distance(p1, p2):
    return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')

def lab_to_rgb(lab):
        # Convert a CIELAB value to an RGB value
        return Color("lab({}% {} {} / 1)".format(*lab)).convert("srgb")