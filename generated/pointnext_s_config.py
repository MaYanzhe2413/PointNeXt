from collections import OrderedDict
import math

def pointnext_s_from_cfg(batch_size=1):
    points = [1024, 512, 256, 128, 64]
    radius = [0.15, 0.22499999999999998, 0.33749999999999997, 0.50625]
    sample_rate = 0.5
    Tree = [True, True, True, True]
    grouped_points = 32
    th = 64
    cfg = OrderedDict([
        ('conv2d0-1', [points[0], 1, 3, 32, 1, 0, 1, batch_size]),  # stem Conv1d
        ('grouper1', [points[0], points[1], batch_size, grouped_points, 0.15, True, th]),
        ('conv2d1-1', [points[1], 1, 32, 64, 1, 0, 1, batch_size]),
        ('relu1-1', []),
        ('conv2d1-2', [points[1], grouped_points, 35, 32, 1, 0, 1, batch_size, sample_rate, True]),
        ('bn1-1', []),
        ('relu1-2', []),
        ('conv2d1-3', [points[1], grouped_points, 32, 64, 1, 0, 1, batch_size]),
        ('bn1-2', []),
        ('maxpool1', [points[1], grouped_points, 64, 1, 0, 1, batch_size]),
        ('grouper2', [points[1], points[2], batch_size, grouped_points, 0.22499999999999998, True, th]),
        ('conv2d2-1', [points[2], 1, 64, 128, 1, 0, 1, batch_size]),
        ('relu2-1', []),
        ('conv2d2-2', [points[2], grouped_points, 67, 64, 1, 0, 1, batch_size, sample_rate, True]),
        ('bn2-1', []),
        ('relu2-2', []),
        ('conv2d2-3', [points[2], grouped_points, 64, 128, 1, 0, 1, batch_size]),
        ('bn2-2', []),
        ('maxpool2', [points[2], grouped_points, 128, 1, 0, 1, batch_size]),
        ('grouper3', [points[2], points[3], batch_size, grouped_points, 0.33749999999999997, True, th]),
        ('conv2d3-1', [points[3], 1, 128, 256, 1, 0, 1, batch_size]),
        ('relu3-1', []),
        ('conv2d3-2', [points[3], grouped_points, 131, 128, 1, 0, 1, batch_size, sample_rate, True]),
        ('bn3-1', []),
        ('relu3-2', []),
        ('conv2d3-3', [points[3], grouped_points, 128, 256, 1, 0, 1, batch_size]),
        ('bn3-2', []),
        ('maxpool3', [points[3], grouped_points, 256, 1, 0, 1, batch_size]),
        ('grouper4', [points[3], points[4], batch_size, grouped_points, 0.50625, True, th]),
        ('conv2d4-1', [points[4], 1, 256, 512, 1, 0, 1, batch_size]),
        ('relu4-1', []),
        ('conv2d4-2', [points[4], grouped_points, 259, 256, 1, 0, 1, batch_size, sample_rate, True]),
        ('bn4-1', []),
        ('relu4-2', []),
        ('conv2d4-3', [points[4], grouped_points, 256, 512, 1, 0, 1, batch_size]),
        ('bn4-2', []),
        ('maxpool4', [points[4], grouped_points, 512, 1, 0, 1, batch_size]),
        ('grouper5', [points[4], points[4], batch_size, grouped_points, None, True, th]),  # GroupAll placeholder
        ('conv2d5-1', [1, points[4], 515, 512, 1, 0, 1, batch_size]),
        ('bn5-1', []),
        ('relu5-1', []),
        ('conv2d5-2', [1, points[4], 512, 512, 1, 0, 1, batch_size]),
        ('bn5-2', []),
        ('relu5-2', []),
        ('maxpool5', [1, points[4], 512, 1, 0, 1, batch_size]),
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