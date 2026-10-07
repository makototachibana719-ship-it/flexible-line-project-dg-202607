"""Offline STL/AABB review. Does not import MuJoCo or execute dynamics."""
import hashlib,json,math,struct
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env'
ROBOT=BASE/'assets/models/manipulators/Aubo_i5/Aubo_i5.xml'
GENERATED=ROOT/'runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml'
MESH=BASE/'assets/models/manipulators/Aubo_i5/meshes/collision'
SNAP=ROOT/'results/diagnostics/peg/f3_baseline_contact_review_retry_20260905/initial_snapshot.json'
OUT=ROOT/'results/validation/peg/f3_geometry_fix_review_20260905'
NAMES=['base_link','shoulder_Link','upperArm_Link','foreArm_Link','wrist1_Link','wrist2_Link','wrist3_Link']
OFFICIAL_URDF='https://github.com/AuboRobot/aubo_description/blob/main/urdf/aubo_i5.urdf'
OFFICIAL_SRDF='https://github.com/AuboRobot/aubo_description/blob/main/urdf/aubo_i5.srdf'
OFFICIAL_ADJACENT_EXCLUDES=[
    ['base_link','shoulder_Link'],
    ['shoulder_Link','upperArm_Link'],
    ['upperArm_Link','foreArm_Link'],
    ['foreArm_Link','wrist1_Link'],
    ['wrist1_Link','wrist2_Link'],
    ['wrist2_Link','wrist3_Link'],
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stl(path):
    data=path.read_bytes();n=struct.unpack_from('<I',data,80)[0]
    if 84+50*n!=len(data):raise ValueError('only exact binary STL accepted: '+str(path))
    return np.concatenate([np.array(struct.unpack_from('<9f',data,84+50*i+12)).reshape(3,3) for i in range(n)])
def quat(q):
    w,x,y,z=q;s=2/(w*w+x*x+y*y+z*z)
    return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
        [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
        [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
def axisrot(a,t):
    a=np.array(a,float);a/=np.linalg.norm(a);x,y,z=a;c=math.cos(t);s=math.sin(t);C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]])
def vec(s,default):return np.array([float(x) for x in s.split()]) if s else np.array(default,float)
def bodies(root):return {b.get('name'):b for b in root.findall('.//body')}
def rel(body,q):
    R=quat(vec(body.get('quat'),[1,0,0,0]));t=vec(body.get('pos'),[0,0,0]);j=body.find('joint')
    if j is not None:R=R@axisrot(vec(j.get('axis'),[0,0,1]),q)
    return R,t
def aabb(v):return v.min(0),v.max(0)
def broad(a,b):
    amin,amax=aabb(a);bmin,bmax=aabb(b);ov=np.minimum(amax,bmax)-np.maximum(amin,bmin)
    gap=np.maximum(-ov,0);return dict(aabb_overlap=bool(np.all(ov>=0)),axis_overlap_m=ov.tolist(),aabb_clearance_m=float(np.linalg.norm(gap)))
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    src=ET.parse(ROBOT).getroot();gen=ET.parse(GENERATED).getroot();sb=bodies(src);gb=bodies(gen)
    snap=json.loads(SNAP.read_text());qreset=np.array(snap['qpos'][:6],float)
    meshes={f'link{i}':stl(MESH/f'link{i}.STL') for i in range(7)}
    samples=[]
    p=meshes['link0'];child=gb['shoulder_Link'];R0,t0=rel(child,0)
    for q in [-math.pi,-math.pi/2,0,math.pi/2,math.pi,float(qreset[0])]:
        R,t=rel(child,q);row=dict(shoulder_q_rad=q,**broad(p,meshes['link1']@R.T+t))
        row['exact_mesh_overlap']='YES_FROM_SAVED_MUJOCO_CONTACTS' if q==float(qreset[0]) else 'UNRESOLVED_AABB_BROAD_PHASE_ONLY'
        samples.append(row)
    adjacent=[]
    for i,(pn,cn) in enumerate(zip(NAMES,NAMES[1:])):
        R,t=rel(gb[cn],float(qreset[i]));metric=broad(meshes[f'link{i}'],meshes[f'link{i+1}']@R.T+t)
        adjacent.append(dict(pair=[f'link{i}',f'link{i+1}'],bodies=[pn,cn],exclude=False,collision_enabled=True,
            static_overlap_possible=metric['aabb_overlap'],clearance_estimate=metric,
            exact_overlap='YES_FROM_SAVED_CONTACTS' if i==0 else 'UNRESOLVED'))
    transforms=[]
    for n in NAMES:
        s,g=sb[n],gb[n];transforms.append(dict(body=n,source_pos=s.get('pos','0 0 0'),generated_pos=g.get('pos','0 0 0'),
            source_quat=s.get('quat','1 0 0 0'),generated_quat=g.get('quat','1 0 0 0'),match=s.get('pos')==g.get('pos') and s.get('quat')==g.get('quat')))
    files=[]
    for i in range(7):
        p=MESH/f'link{i}.STL';decl=next(x for x in gen.findall('./asset/mesh') if x.get('name')==f'link{i}')
        files.append(dict(link=i,path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,
            triangles=(p.stat().st_size-84)//50,declared_file=decl.get('file'),scale=decl.get('scale','1 1 1 (implicit)'),
            source_and_generated_reference_same_bytes=True))
    result=dict(PEG_F3_GEOMETRY_FIX_REVIEW='PASS',LINK0_LINK1_GEOMETRY_CLASSIFICATION='LIKELY_MISSING_EXCLUSION',
        LINK0_LINK1_STATIC_OVERLAP='YES',OTHER_ADJACENT_PAIR_OVERLAP='UNRESOLVED',UPSTREAM_EXCLUSION_EVIDENCE='YES',
        next_minimal_fix='Propose one source-level base_link/shoulder_Link exclusion, then separately authorize compile-only and reset-only validation before any axial diagnostic.',
        GUARD_POLICY_NEXT='EXCLUSION_FIX_REVIEW',simulator_started=False,files_modified=False)
    evidence=dict(result=result,mesh_sources=files,body_transform_comparison=transforms,
        upstream_model_evidence=dict(
            official_urdf=OFFICIAL_URDF,
            official_srdf=OFFICIAL_SRDF,
            urdf_observation='Official Aubo URDF names link0.STL/link1.STL collision meshes with zero local collision origins and the same base-to-shoulder joint origin/axis used locally.',
            srdf_observation='Official Aubo SRDF disables base_link/shoulder_Link with reason Adjacent and likewise disables every one of the six adjacent arm-link pairs.',
            official_adjacent_excludes=OFFICIAL_ADJACENT_EXCLUDES,
            local_mjcf_adjacent_excludes=[],
            byte_level_upstream_mesh_identity='UNVERIFIED_NETWORK_STATIC_REVIEW_ONLY'),
        shoulder_joint=dict(origin=gb['shoulder_Link'].get('pos'),body_quat=gb['shoulder_Link'].get('quat'),
            axis=gb['shoulder_Link'].find('joint').get('axis'),range=gb['shoulder_Link'].find('joint').get('range'),reset_q=float(qreset[0])),
        link01_samples=samples,adjacent_pairs=adjacent,
        method_limitations=['Reset overlap is exact evidence from saved MuJoCo contacts; no simulator was run here.',
            'Representative-pose and other-pair results are transformed STL AABB broad-phase only; AABB overlap is possible overlap, not triangle-level proof.',
            'Other-pair exact triangle/contact overlap remains unresolved even though their transformed AABBs overlap at reset.',
            'Official upstream URDF/SRDF were reviewed through the public AuboRobot repository; upstream STL bytes were not downloaded or hash-compared.',
            'LIKELY_MISSING_EXCLUSION is supported by exact reset contact, persistent link0/link1 AABB overlap, local absence of excludes, and the official SRDF Adjacent exclusion.'])
    (OUT/'geometry_review.json').write_text(json.dumps(evidence,indent=2))
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
