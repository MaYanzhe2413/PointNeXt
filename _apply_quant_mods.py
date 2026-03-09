#!/usr/bin/env python3
"""Script to apply quantization modifications to pointnext.py"""
import os
os.chdir('/workspace/PointNeXt')

with open('openpoints/models/backbone/pointnext.py', 'r') as f:
    content = f.read()

replacements = []

# 1. Imports + get_reduction_fn
replacements.append((
    'from typing import List, Type\n'
    'import logging\n'
    'import torch\n'
    'import torch.nn as nn\n'
    'from ..build import MODELS\n'
    'from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \\\n'
    '    create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres, kdtree_sample\n'
    '\n\n'
    'def get_reduction_fn(reduction):\n'
    "    reduction = 'mean' if reduction.lower() == 'avg' else reduction\n"
    "    assert reduction in ['sum', 'max', 'mean']\n"
    "    if reduction == 'max':\n"
    '        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]\n'
    "    elif reduction == 'mean':\n"
    '        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)\n'
    "    elif reduction == 'sum':\n"
    '        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)\n'
    '    return pool',
    # NEW
    'from typing import List, Type\n'
    'import logging\n'
    'import torch\n'
    'import torch.nn as nn\n'
    'import torch.ao.quantization as quant\n'
    'from ..build import MODELS\n'
    'from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \\\n'
    '    create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres, kdtree_sample\n'
    'from ..layers.quant_utils import (\n'
    '    MaxPool, MeanPool, SumPool, get_reduction_module,\n'
    '    QAdd, QCat,\n'
    ')\n'
    '\n\n'
    'def get_reduction_fn(reduction):\n'
    '    """Legacy wrapper. Returns nn.Module instead of lambda."""\n'
    '    return get_reduction_module(reduction)',
))

# 2. LocalAggregation pool
replacements.append((
    '        self.pool = get_reduction_fn(self.reduction)',
    '        self.pool = get_reduction_module(self.reduction)',
))

# 3. SetAbstraction pool lambda -> MaxPool + qadd
replacements.append((
    '            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]',
    '            self.pool = MaxPool()\n'
    '            if self.use_res:\n'
    '                self.qadd = QAdd()',
))

# 4. Remove DEBUG block in SetAbstraction forward
replacements.append((
    '                new_p = p\n'
    '            """ DEBUG neighbor numbers. \n'
    '            query_xyz, support_xyz = new_p, p\n'
    '            radius = self.grouper.radius\n'
    '            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())\n'
    '            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])\n'
    "            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')\n"
    '            DEBUG end """\n'
    "            if self.use_res or 'df' in self.feature_type:",
    '                new_p = p\n'
    "            if self.use_res or 'df' in self.feature_type:",
))

# 5. SA forward: f + identity -> qadd
replacements.append((
    '            if self.use_res:\n'
    '                f = self.act(f + identity)',
    '            if self.use_res:\n'
    '                f = self.act(self.qadd(f, identity))',
))

# 6a. FP init: store upsample
replacements.append((
    '        super().__init__()\n'
    '        if not upsample:',
    '        super().__init__()\n'
    '        self.upsample = upsample\n'
    '        if not upsample:',
))

# 6b. FP: add QCat in non-upsample branch
replacements.append((
    '            self.linear1 = nn.Sequential(*linear1)\n'
    '        else:',
    '            self.linear1 = nn.Sequential(*linear1)\n'
    '            self.qcat = QCat(dim=1)\n'
    '        else:',
))

# 6c. FP: add QCat in upsample branch + replace lambda pool
replacements.append((
    '            self.convs = nn.Sequential(*convs)\n'
    '\n'
    '        self.pool = lambda x: torch.m
    '            self.
    '            self.qcat = QCat(dim=1)\n'
    '\n'
    '      
))

# 7a.
replacements.append((
    '     

    "                (f, self.linear2(f_global).unsqueeze(-1).e
    '            f_glo

))

# 7b. FP forw

    '                f = self.c
    '                
    '                f = self.convs(\
    '                 
))

# 8. InvResMLP: 
replacements.append((
    '  
    '        self.use_res = use_re
    '        mid_channels = int(in_channels * expansion)',
    '        super().__init__()\n'
  
    '        self.qadd = QAdd() if use_res else None\n'
    '        mid_channels = int(in_channels * expansion)',
))

# 9. In
replacemen
    '        f = self.pwconv(f)\n'
    '
    '            f += identity',
    '        f = self.pwconv(f)\n'
    '  
    '            f = self.qadd(f, identity)',
))

# 10
replacements.append((
    '       
    '        mid_channels = in_channels * expansion\n'
    '        sel
    '        self.use_res = use_res\n
    '        self.qadd = QAdd() if use_res else None\n'
    '        
    '        self.convs = LocalAggre
))

# 11. ResBlock forward: f += identity -> qadd
replacements.appen
    '        f = self.
    '    
    '            f += identity\n'
   
    '        return [p
    '\n\n'
    '@MOD
    'class PointNextEncoder',
    '        f = self.convs([p, f])\n'
    ' 
    '            f = self.qadd(f, 
    '        f = self.act(f)\n'
    '        return [p, f]\n'
    '\n\n'
    '@MODELS.register_m
    'class PointNextEncoder',
))

for i, (old, new) in enumerate(replacements):
    if old in content:
        content = conte
        print(f"  [{i+1}] OK")
    else:
 
        # Show first 80 chars of what we're looking for
     

with open('openpoints/
    f.wri

print("\n
