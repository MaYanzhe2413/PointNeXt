from collections import OrderedDict

def pnn_config_from_json(batch_size=1):
    pointnet = OrderedDict([
        ('fps_stage', [64, 16, 3, 0, 1000, 2000, 3000]),  # FPS: N, K, C, offsets
        ('__points_after_fps_stage__', [16]),  # helper: points after FPS
        ('mlp_stage_conv1', [__points_after_fps_stage__[0] if 'fps_stage' in locals() else 1, 1, 3, 8, 1, 0, 1, batch_size]),  # Conv2D 1x1
        ('mlp_stage_meminfo', [0, 5000, 8000]),  # ifmap/filter/ofmap offsets
    ])
    return pointnet