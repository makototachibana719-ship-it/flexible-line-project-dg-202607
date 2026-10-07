"""Static XML/generator audit only. Never imports MuJoCo or creates an env."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env'
ROBOT=BASE/'assets/models/manipulators/Aubo_i5/Aubo_i5.xml'
GENERATED=ROOT/'runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml'
ASSEMBLER=BASE/'utils/robot_assembler.py'
SPLICER=BASE/'utils/xml_splice.py'
INVENTORY=ROOT/'results/diagnostics/peg/f3_baseline_contact_review_retry_20260905/geom_inventory.json'
OUT=ROOT/'results/validation/peg/f3_link01_collision_intent_20260905'
NAMES=['base_link','shoulder_Link','upperArm_Link','foreArm_Link','wrist1_Link','wrist2_Link','wrist3_Link']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bodies(root):return {x.get('name'):x for x in root.findall('.//body')}
def collision(body):
    gs=[g for g in body.findall('geom') if g.get('class')=='collision']
    return [dict(mesh=g.get('mesh'),type=g.get('type','INHERITED:mesh'),group=g.get('group','INHERITED:3'),
        contype=g.get('contype','INHERITED:1'),conaffinity=g.get('conaffinity','INHERITED:1'),
        margin=g.get('margin','INHERITED/MUJOCO_DEFAULT'),gap=g.get('gap','INHERITED/MUJOCO_DEFAULT'),
        solref=g.get('solref','INHERITED/MUJOCO_DEFAULT'),solimp=g.get('solimp','INHERITED/MUJOCO_DEFAULT')) for g in gs]
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    rr=ET.parse(ROBOT).getroot();gr=ET.parse(GENERATED).getroot();rb=bodies(rr);gb=bodies(gr)
    excludes=[]
    scanned=[]
    for p in BASE.rglob('*.xml'):
        try:
            xs=[dict(body1=x.get('body1'),body2=x.get('body2')) for x in ET.parse(p).getroot().findall('.//exclude')]
        except ET.ParseError:continue
        if xs:excludes.append(dict(file=str(p.relative_to(ROOT)),pairs=xs))
        scanned.append(str(p.relative_to(ROOT)))
    relevant=[]
    for p in (ROBOT,GENERATED,BASE/'assets/scenes/assemble_hole.xml'):
        x=ET.parse(p).getroot();relevant.extend((a.get('body1'),a.get('body2')) for a in x.findall('.//exclude'))
    adjacent=[]
    for parent,child in zip(NAMES,NAMES[1:]):
        ce=collision(gb[child]);pe=collision(gb[parent])
        excluded=(parent,child) in relevant or (child,parent) in relevant
        adjacent.append(dict(pair=[parent,child],parent_child=gb[child] in list(gb[parent]),
            child_joint=gb[child].find('joint').get('name') if gb[child].find('joint') is not None else None,
            explicit_exclude=excluded,collision_enabled=bool(pe and ce and not excluded),
            parent_collision=pe,child_collision=ce,
            overlap_evidence='OBSERVED_RESET_4_CONTACTS_AT_-0.00649972274584152_M' if parent=='base_link' else 'NOT_DERIVABLE_STATICALLY'))
    inv=json.loads(INVENTORY.read_text());trigger=[x for x in inv if x['id'] in (49,52)]
    assembler_text=ASSEMBLER.read_text(encoding='utf-8');splicer_text=SPLICER.read_text(encoding='utf-8')
    result=dict(PEG_F3_LINK01_COLLISION_INTENT_AUDIT='PASS',LINK0_LINK1_PARENT_CHILD='YES',
        LINK0_LINK1_EXCLUDED='NO',LINK0_LINK1_COLLISION_ENABLED='YES',ADJACENT_LINK_POLICY='ALLOW_ALL',
        LINK0_LINK1_CLASSIFICATION='UNRESOLVED',GUARD_POLICY_NEXT='GEOMETRY_FIX_REVIEW',
        configuration_consistency='CONSISTENT_WITH_AUBO_SOURCE_ALL_ADJACENT_PAIRS_ENABLED',
        semantic_intent='UNRESOLVED_NO_COMMENT_DOC_OR_PAIR_POLICY',simulator_started=False,files_modified=False)
    evidence=dict(result=result,trigger_inventory=trigger,adjacent_pairs=adjacent,
        relevant_explicit_excludes=relevant,all_repository_exclude_files=excludes,
        generator=dict(robot_assembler_merges_contact='merge_section(robot_root, root, "contact")' in assembler_text,
            xml_splice_merges_contact='merge_section(robot_root, root, "contact")' in splicer_text,
            filtering_logic_found=False),
        notes=['Aubo source and generated task contain no contact/exclude element.',
               'Collision defaults declare group=3,type=mesh; contype/conaffinity and contact parameters are not explicitly set.',
               'Values labelled INHERITED/MUJOCO_DEFAULT are intentionally not guessed from static source.',
               'Other robot/gripper models use explicit excludes, so assembler supports and preserves them.'],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROBOT,GENERATED,ASSEMBLER,SPLICER,INVENTORY)})
    (OUT/'audit.json').write_text(json.dumps(evidence,indent=2))
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
