from collections import OrderedDict
import math

def point_config_from_model(batch_size=1):
    points = [1024, 512, 128, 1]
    grouped_points_default = 32
    cfg = OrderedDict([
        ('grouper1', [points[0], points[1], batch_size, 32, 0.2]),
        ('conv2d1-1', [points[1], 32, 6, 64, 1, 0, 1, batch_size]),
        ('bn1-1', []),
        ('relu1-1', []),
        ('conv2d1-2', [points[1], 32, 64, 64, 1, 0, 1, batch_size]),
        ('bn1-2', []),
        ('relu1-2', []),
        ('conv2d1-3', [points[1], 32, 64, 128, 1, 0, 1, batch_size]),
        ('bn1-3', []),
        ('relu1-3', []),
        ('maxpool1', [points[1], 32, 128, 1, 0, 1, batch_size]),
        ('grouper2', [points[1], points[2], batch_size, 64, 0.4]),
        ('conv2d2-1', [points[2], 64, 131, 128, 1, 0, 1, batch_size]),
        ('bn2-1', []),
        ('relu2-1', []),
        ('conv2d2-2', [points[2], 64, 128, 128, 1, 0, 1, batch_size]),
        ('bn2-2', []),
        ('relu2-2', []),
        ('conv2d2-3', [points[2], 64, 128, 256, 1, 0, 1, batch_size]),
        ('bn2-3', []),
        ('relu2-3', []),
        ('maxpool2', [points[2], 64, 256, 1, 0, 1, batch_size]),
        ('grouper3', [points[2], points[3], batch_size, points[2], None]),  # GroupAll
        ('conv2d3-1', [points[3], points[2], 259, 256, 1, 0, 1, batch_size]),
        ('bn3-1', []),
        ('relu3-1', []),
        ('conv2d3-2', [points[3], points[2], 256, 512, 1, 0, 1, batch_size]),
        ('bn3-2', []),
        ('relu3-2', []),
        ('conv2d3-3', [points[3], points[2], 512, 1024, 1, 0, 1, batch_size]),
        ('bn3-3', []),
        ('relu3-3', []),
        ('maxpool3', [points[3], points[2], 1024, 1, 0, 1, batch_size]),
        ('linear1-1', [1024, 512, 1, batch_size]),
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