"""Behavioral tests for bounded home-aware IK; no ROS or robot services."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experimental_pinocchio_pyroki'))
from continuous_ik import ContinuousIK,minimum_jerk_transition


def make_model(path):
    origins=[('0 0 .333','0 0 0'),('0 0 0','-1.5707963267948966 0 0'),
        ('0 -.316 0','1.5707963267948966 0 0'),('.0825 0 0','1.5707963267948966 0 0'),
        ('-.0825 .384 0','-1.5707963267948966 0 0'),('0 0 0','1.5707963267948966 0 0'),
        ('.088 0 0','1.5707963267948966 0 0')]
    lo=[-2.9,-1.76,-2.9,-3.07,-2.87,.44,-3.05];hi=[2.9,1.76,2.9,-.12,2.87,4.5,3.05]
    xml=['<robot name="synthetic_dual"><link name="root"/>']
    for side,mount in [('left','0 -.2 0'),('right','0 .2 0')]:
        prefix=side+'_fr3v2_'
        xml.append(f'<link name="{prefix}link0"/><joint name="{side}_mount" type="fixed"><parent link="root"/><child link="{prefix}link0"/><origin xyz="{mount}"/></joint>')
        for i,(xyz,rpy) in enumerate(origins,1):
            xml.append(f'<link name="{prefix}link{i}"/><joint name="{prefix}joint{i}" type="revolute"><parent link="{prefix}link{i-1}"/><child link="{prefix}link{i}"/><origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="0 0 1"/><limit lower="{lo[i-1]}" upper="{hi[i-1]}" effort="50" velocity="2"/></joint>')
        xml.append(f'<link name="{prefix}link8"/><joint name="{prefix}joint8" type="fixed"><parent link="{prefix}link7"/><child link="{prefix}link8"/><origin xyz="0 0 .107"/></joint>')
    path.write_text(''.join(xml)+'</robot>')


@unittest.skipUnless(importlib.util.find_spec('pinocchio') and importlib.util.find_spec('scipy'),'Optional IK dependencies unavailable')
class HomeIKTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.urdf=self.root/'model.urdf';make_model(self.urdf)
        self.left=ContinuousIK(self.urdf,'left_fr3v2_link8',range(7))
        self.right=ContinuousIK(self.urdf,'right_fr3v2_link8',range(7,14))
        self.home=np.tile([.2,.4,-.3,-1.8,.3,2.5,.2],2)
        self.names=list(self.left.model.names)[1:]

    def test_pose_equality_step_bound_and_inactive_arm(self):
        desired=self.home.copy();desired[:7]+=[.02,-.01,.01,0,.02,-.01,.01]
        target=self.left.fk(desired)
        q,result=self.left.solve_near(target,self.home,max_step=.08)
        self.assertTrue(result.converged);self.assertLessEqual(np.max(abs(q-self.home)),.08+1e-9)
        np.testing.assert_array_equal(q[7:],self.home[7:])
        actual=self.left.fk(q)
        self.assertLess(np.linalg.norm(actual[:3,3]-target[:3,3]),1e-5)
        self.assertLess(np.linalg.norm(actual[:3,:3]-target[:3,:3]),2e-4)
        with self.assertRaises(RuntimeError):self.left.solve_near(target,self.home,max_step=1e-5)
        invalid=target.copy();invalid[0,0]+=1
        with self.assertRaises(ValueError):self.left.solve_near(invalid,self.home)

    def fixture(self):
        samples=[]
        for i in range(3):
            q=self.home.copy();q[[0,7]]+=.02+.01*i
            s={'t_s':float(i)*.05}
            for j,(side,solver) in enumerate([('left',self.left),('right',self.right)]):
                root_target=solver.fk(q)
                base=np.eye(4);base[1,3]=-.2 if side=='left' else .2
                s[side]={'base_T_link8_target':{'matrix':(np.linalg.inv(base)@root_target).tolist()},
                    'source_arm_joint_positions_rad':q[j*7:(j+1)*7].tolist(),'hand_joint_positions_rad':[i*.01]*22}
            samples.append(s)
        plan={'execution_enabled':False,'output_controlled_frame':'fr3_link8','samples':samples}
        source=self.root/'plan.json';source.write_text(json.dumps(plan))
        state=self.root/'state.json';state.write_text(json.dumps({'execution_enabled':False,'state_kind':'synthetic_test',
            'joint_names':self.names,'joint_positions_rad':self.home.tolist()}))
        return source,state,plan

    def invoke(self,source,state,output,*extra):
        entry = 'solve_continuous_plan.py' if state or extra else 'solve_bimanual_plan.py'
        command=[sys.executable,str(ROOT/'experimental_pinocchio_pyroki'/entry),str(source),str(output),'--urdf',str(self.urdf)]
        if state:command+=['--start-state',str(state)]
        return subprocess.run(command+list(extra),capture_output=True,text=True,timeout=20)

    def test_complete_replay_uses_start_state_preserves_targets_times_and_hands(self):
        source,state,plan=self.fixture();output=self.root/'new.json'
        result=self.invoke(source,state,output);self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(output.read_text());transition=json.loads((self.root/'new_start_transition.json').read_text())
        self.assertFalse(data['execution_enabled']);self.assertEqual(len(data['samples']),3)
        self.assertEqual(data['ik_summary']['initial_seed_joint_positions_rad'],self.home.tolist())
        self.assertEqual(transition['samples'][0]['joint_positions_rad'],self.home.tolist())
        first=data['samples'][0]['left_arm_joint_positions_rad']+data['samples'][0]['right_arm_joint_positions_rad']
        self.assertEqual(transition['samples'][-1]['joint_positions_rad'],first)
        for old,new in zip(plan['samples'],data['samples']):
            self.assertEqual(old['t_s'],new['t_s'])
            for side in ['left','right']:self.assertEqual(old[side]['hand_joint_positions_rad'],new[side+'_hand_joint_positions_rad'])
        legacy=self.root/'legacy.json';result=self.invoke(source,None,legacy)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_single_frame_is_marked_and_failure_never_exports_partial_replay(self):
        source,state,plan=self.fixture();output=self.root/'first.json'
        result=self.invoke(source,state,output,'--first-frame-only');self.assertEqual(result.returncode,0,result.stderr)
        doc=json.loads(output.read_text());self.assertEqual(doc['artifact_type'],'offline_home_aware_frame0_ik')
        self.assertEqual(len(doc['samples']),1);self.assertEqual(doc['input_frame_count'],3)
        plan['samples'][1]['right']['base_T_link8_target']['matrix'][0][3]=10.
        source.write_text(json.dumps(plan));failed=self.root/'failed.json'
        result=self.invoke(source,state,failed);self.assertNotEqual(result.returncode,0)
        self.assertFalse(failed.exists());self.assertFalse((self.root/'failed_start_transition.json').exists())
        error=json.loads(failed.with_suffix('.failure.json').read_text());self.assertEqual(error['failed_frame'],1)
        self.assertFalse(error['execution_enabled']);self.assertFalse(error['output_written'])

    def test_outputs_cannot_overwrite_input(self):
        source,state,_=self.fixture();original=source.read_bytes()
        result=self.invoke(source,state,source)
        self.assertNotEqual(result.returncode,0);self.assertEqual(source.read_bytes(),original)
        self.assertIn('must not overwrite',result.stderr)


class TransitionTests(unittest.TestCase):
    def test_analytic_caps_endpoints_and_invalid_inputs(self):
        start=np.zeros(14);goal=np.linspace(-1.2,1.2,14);lo=np.full(14,-2.);hi=-lo
        t,q,v,a,r=minimum_jerk_transition(start,goal,lo,hi)
        np.testing.assert_array_equal(q[0],start);np.testing.assert_array_equal(q[-1],goal)
        np.testing.assert_allclose(v[[0,-1]],0,atol=1e-14);np.testing.assert_allclose(a[[0,-1]],0,atol=1e-14)
        self.assertLessEqual(r['peak_velocity_rad_s'],.09);self.assertLessEqual(r['peak_acceleration_rad_s2'],.108)
        self.assertLessEqual(r['peak_jerk_rad_s3'],.432)
        self.assertTrue(np.all(q>=lo) and np.all(q<=hi));self.assertFalse(r['execution_enabled'])
        for kwargs in [{'velocity':float('nan')},{'dt':0.},{'jerk':-1.}]:
            with self.assertRaises(ValueError):minimum_jerk_transition(start,goal,lo,hi,**kwargs)
        with self.assertRaises(ValueError):minimum_jerk_transition(start,goal*10,lo,hi)


if __name__=='__main__':unittest.main()
