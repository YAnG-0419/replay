"""Convert an explicitly selected named home into an OFFLINE IK start state."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def convert(path,label,variant):
    import yaml
    source=yaml.safe_load(path.read_text())
    matches=[(key,t) for key,t in source['tasks'].items() if t.get('label')==label]
    if len(matches)!=1:raise ValueError('Home label must select exactly one task')
    key,task=matches[0];names=[];values=[];mapping={}
    if variant not in ['fr3','fr3v2']:raise ValueError('Expected explicit fr3 or fr3v2 joint variant')
    for side in ['left','right']:
        arm=task['home'][side]
        if arm is None:raise ValueError('Both arm home poses are required; missing poses are not filled with zeros')
        targets=[f'{side}_{variant}_joint{i}' for i in range(1,8)]
        source_names=arm['joint_names']
        if source_names not in [[f'{side}_{v}_joint{i}' for i in range(1,8)] for v in ['fr3','fr3v2']]:raise ValueError('Unexpected side or joint order')
        q=np.asarray(arm['positions'],dtype=float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Expected seven finite joint angles')
        names.extend(targets);values.extend(q.tolist());mapping.update(dict(zip(source_names,targets)))
    return {'execution_enabled':False,'robot_profile':'franka-sharpa','state_kind':'named_home_target_not_live_measurement','angle_unit':'radian',
        'pose_label':label,'pose_key':key,'joint_names':names,'joint_positions_rad':values,'explicit_joint_name_mapping':mapping,
        'source_path':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('poses',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--label',required=True);p.add_argument('--model-joint-variant',choices=['fr3','fr3v2'],required=True)
    p.add_argument('--angle-unit',choices=['radian'],required=True,help='Explicitly confirm source positions use radians')
    a=p.parse_args();document=convert(a.poses,a.label,a.model_joint_variant)
    if a.output.resolve()==a.poses.resolve():raise ValueError('Do not overwrite source poses')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(document,indent=2)+'\n')
    print(f'Wrote offline {a.label} start state to {a.output}; not live robot feedback')


if __name__=='__main__':main()
