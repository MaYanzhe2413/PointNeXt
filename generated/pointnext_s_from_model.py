from collections import OrderedDict
import math

def point_config_from_model(batch_size=1):
    points = [1024, 512, 256, 128, 64, 1]
    grouped_points_default = 32
    th = 64
    cfg = OrderedDict([
        ('grouper1', [points[0], points[1], batch_size, 32, 0.15, True, th]),
        ('conv2d1-1', [points[1], 32, 6, 32, 1, 0, 1, batch_size]),
        ('bn1-1', []),
        ('relu1-1', []),
        ('conv2d1-2', [points[1], 32, 32, 64, 1, 0, 1, batch_size]),
        ('bn1-2', []),
        ('relu1-2', []),
        ('maxpool1', [points[1], 32, 64, 1, 0, 1, batch_size]),
        ('grouper2', [points[1], points[2], batch_size, 32, 0.22499999999999998, True, th]),
        ('conv2d2-1', [points[2], 32, 67, 64, 1, 0, 1, batch_size]),
        ('bn2-1', []),
        ('relu2-1', []),
        ('conv2d2-2', [points[2], 32, 64, 128, 1, 0, 1, batch_size]),
        ('bn2-2', []),
        ('relu2-2', []),
        ('maxpool2', [points[2], 32, 128, 1, 0, 1, batch_size]),
        ('grouper3', [points[2], points[3], batch_size, 32, 0.33749999999999997, True, th]),
        ('conv2d3-1', [points[3], 32, 131, 128, 1, 0, 1, batch_size]),
        ('bn3-1', []),
        ('relu3-1', []),
        ('conv2d3-2', [points[3], 32, 128, 256, 1, 0, 1, batch_size]),
        ('bn3-2', []),
        ('relu3-2', []),
        ('maxpool3', [points[3], 32, 256, 1, 0, 1, batch_size]),
        ('grouper4', [points[3], points[4], batch_size, 32, 0.50625, True, th]),
        ('conv2d4-1', [points[4], 32, 259, 256, 1, 0, 1, batch_size]),
        ('bn4-1', []),
        ('relu4-1', []),
        ('conv2d4-2', [points[4], 32, 256, 512, 1, 0, 1, batch_size]),
        ('bn4-2', []),
        ('relu4-2', []),
        ('maxpool4', [points[4], 32, 512, 1, 0, 1, batch_size]),
        ('grouper5', [points[4], points[5], batch_size, points[4], None, True, th]),  # GroupAll
        ('conv2d5-1', [points[5], points[4], 515, 512, 1, 0, 1, batch_size]),
        ('bn5-1', []),
        ('relu5-1', []),
        ('conv2d5-2', [points[5], points[4], 512, 512, 1, 0, 1, batch_size]),
        ('bn5-2', []),
        ('relu5-2', []),
        ('maxpool5', [points[5], points[4], 512, 1, 0, 1, batch_size]),
        ('linear1-1', [512, 512, 1, batch_size]),
        ('bn6-1', []),
        ('relu6-1', []),
        ('dropout1', []),
        ('linear2-1', [512, 256, 1, batch_size]),
        ('bn7-1', []),
        ('relu7-1', []),
        ('dropout2', []),
        ('linear3-1', [256, 40, 1, batch_size]),
    ])
    return cfg