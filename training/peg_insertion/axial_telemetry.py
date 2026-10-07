"""Opt-in passive observer. No controller monkeypatch, extra solve or mj_step.

Usage (only in a separately authorized diagnostic):
    observer = AxialTelemetry(adapter, new_output_directory)
    result = observer.step(action)  # exactly one original adapter.step
    observer.close()

Python profiling copies already-computed values; it does not replace functions.
Profiling is scoped to the calling thread and restored even on exceptions.
"""
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import numpy as np

V2='PEG_OBS_CONTRACT_V2_SENSOR_FIXED'
FAILURE_STAGES=('RAW_STEP','IK_CAPTURE','JOINT_CAPTURE','ACTUATOR_CAPTURE','CONTACT_CAPTURE',
                'POSTPROCESS','JSON_NORMALIZE','SERIALIZE','WRITE','FSYNC','PUBLISH')
LAST_SUCCESSFUL_STAGE={'POSTPROCESS':'CONTACT_CAPTURE','JSON_NORMALIZE':'POSTPROCESS','SERIALIZE':'JSON_NORMALIZE',
                       'WRITE':'SERIALIZE','FSYNC':'WRITE','PUBLISH':'FSYNC'}

class TelemetryStageError(RuntimeError):
    def __init__(self,stage,original):
        super().__init__(str(original));self.stage=stage;self.original=original

def normalize_json(value):
    """Strict value-preserving JSON normalization; unknown objects fail closed."""
    if isinstance(value,np.ndarray):return normalize_json(value.tolist())
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.floating):return float(value)
    if dataclasses.is_dataclass(value):return {f.name:normalize_json(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if value is None or type(value) in (bool,int,float,str):return value
    if isinstance(value,dict):
        if any(type(key) is not str for key in value):raise TypeError('JSON object keys must be strings')
        return {key:normalize_json(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [normalize_json(item) for item in value]
    raise TypeError('Unsupported telemetry JSON type: '+type(value).__name__)

def copy_value(value):
    if isinstance(value,np.ndarray):return value.copy().tolist()
    if isinstance(value,np.generic):return value.item()
    if dataclasses.is_dataclass(value):return {f.name:copy_value(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,(tuple,list)):return [copy_value(v) for v in value]
    if isinstance(value,dict):return {str(k):copy_value(v) for k,v in value.items()}
    if hasattr(value,'name'):return value.name
    if value is None or type(value) in (str,int,float,bool):return value
    raise TypeError('Unsupported telemetry capture type: '+type(value).__name__)

def code_of(fn):return getattr(getattr(fn,'__func__',fn),'__code__',None)

class AxialTelemetry:
    def __init__(self,env,output=None,*,enabled=True,solver=None,contact_force_reader=None):
        self.env=env;self.enabled=enabled;self.stream=None;self.output=None;self.count=0;self.last=None
        self.phase=None;self.failure_evidence_complete=False
        if not enabled:return
        if sys.getprofile() is not None:raise RuntimeError('existing profiler: refuse to replace')
        if env.observation_contract_version!=V2:raise ValueError('V2 contract required')
        self.raw=env.env;self.controller=self.raw.controller
        if solver is None:
            from mujoco.minimize import least_squares
            solver=least_squares
        self.solver=solver
        if contact_force_reader is None:
            from mujoco import mj_contactForce
            contact_force_reader=mj_contactForce
        self.contact_force_reader=contact_force_reader
        self.codes={name:code_of(getattr(self.controller,name)) for name in
                    ('solve_ik','compute_control','_ik_jacobian','compute_jacobian')}
        self.codes['least_squares']=code_of(solver);self.codes['_set_action']=code_of(self.raw._set_action)
        if any(v is None for v in self.codes.values()):raise ValueError('unsupported non-Python observer boundary')
        files={Path(inspect.getsourcefile(type(self.controller))),Path(inspect.getsourcefile(type(self.raw))),
               Path(inspect.getsourcefile(type(env))),Path(inspect.getsourcefile(solver)),Path(__file__)}
        self.binding={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if output is not None:
            path=Path(output);path.mkdir(parents=True,exist_ok=False)
            self.output=path
            (path/'binding.json').write_text(json.dumps(dict(schema='PEG_AXIAL_TELEMETRY_V1',files=self.binding,
                observation_contract_version=V2,contacts_phase='pre_step and post_physics; not every physics substep',
                jacobian_scope='existing evaluations only; q attached; last evaluated may differ from returned q',
                training_data=False),indent=2))
            self.stream=(path/'steps.jsonl').open('x')

    def set_context(self,*,phase):
        """Attach caller-known diagnostic context without touching the environment."""
        if not isinstance(phase,str) or not phase:raise ValueError('non-empty phase required')
        self.phase=phase

    @staticmethod
    def _sync_directory(path):
        if os.name=='posix':
            fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)

    def _failure_file_fsync(self,fd):os.fsync(fd)
    def _failure_publish(self,temp_path,final_path):os.link(temp_path,final_path)
    def _step_fsync(self):os.fsync(self.stream.fileno())
    def _singular_values(self,j):return np.linalg.svd(j,compute_uv=False).tolist()
    def _normalize_row(self,row):return normalize_json(row)
    def _serialize_row(self,row):return json.dumps(row,allow_nan=False)+'\n'
    def _write_step_payload(self,payload):self.stream.write(payload);self.stream.flush()

    def _durable_failure(self,artifact):
        """Serialize first, then fsync and atomically publish a no-clobber artifact."""
        if self.output is None:raise RuntimeError('failure evidence output is not configured')
        final=self.output/f"failure_step_{self.count:06d}.json"
        stage='JSON_NORMALIZE';tmp=None
        try:
            normalized=normalize_json(artifact)
            stage='SERIALIZE';payload=json.dumps(normalized,indent=2,allow_nan=False).encode('utf-8')
            stage='WRITE';fd,tmp=tempfile.mkstemp(prefix='.'+final.name+'.',suffix='.tmp',dir=self.output)
            with os.fdopen(fd,'wb') as stream:
                stream.write(payload);stream.flush();stage='FSYNC';self._failure_file_fsync(stream.fileno())
            stage='PUBLISH';self._failure_publish(tmp,final)
            self._sync_directory(self.output)
            return final
        except Exception as exc:
            raise TelemetryStageError(stage,exc) from exc
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)
                try:self._sync_directory(self.output)
                except Exception:pass

    @staticmethod
    def _exception_fields(exc,tb=None):
        original=exc.original if isinstance(exc,TelemetryStageError) else exc
        if tb is None:summary=traceback.format_exception(type(original),original,original.__traceback__)
        else:summary=traceback.format_exception(type(original),original,tb)
        return dict(exception_type=type(original).__name__,exception_message=str(original),
                    traceback_summary=''.join(summary[-8:]))

    def _availability(self,row):
        pre=row.get('pre',{});post=row.get('post',{})
        return dict(raw_step=bool(row.get('_raw_step_available',False)),
                    ik=bool(row.get('ik')) and bool(row.get('solver')),
                    joint=all('joints' in item for item in (pre,post)),
                    actuator=all('actuators' in item for item in (pre,post)) and
                             all(key in row for key in ('controller_output','ctrl_before_write','ctrl_after_write')),
                    contact=all('contacts' in item for item in (pre,post)))

    @staticmethod
    def _missing_fields(row):
        required=('ik','solver','controller_output','ctrl_before_write','ctrl_after_write','pre','post')
        missing=[key for key in required if key not in row]
        for side in ('pre','post'):
            for key in ('joints','actuators','contacts'):
                if key not in row.get(side,{}):missing.append(side+'.'+key)
        return missing

    @staticmethod
    def _stage_for_missing(missing):
        if any(key in missing for key in ('ik','solver')):return 'IK_CAPTURE'
        if any('joint' in key for key in missing):return 'JOINT_CAPTURE'
        if any(key in missing for key in ('controller_output','ctrl_before_write','ctrl_after_write')) or any('actuator' in key for key in missing):return 'ACTUATOR_CAPTURE'
        if any('contact' in key for key in missing):return 'CONTACT_CAPTURE'
        return 'POSTPROCESS'

    def _record_failure(self,stage,last_successful,exc,missing,tb=None):
        if stage not in FAILURE_STAGES:stage='POSTPROCESS'
        artifact=dict(step_index=self.count,phase=self.phase,failure_stage=stage,
            last_successful_stage=last_successful,**self._exception_fields(exc,tb),
            availability=self._availability(self._row),missing_fields=list(missing),complete=False)
        self._row['complete']=False;self.last=self._row;self.failure_evidence_complete=False
        try:
            path=self._durable_failure(artifact)
            self.failure_evidence_complete=True
            self._row['failure_evidence_durable']=True;self._row['failure_artifact']=path.name
        except Exception as persist_exc:
            if not isinstance(persist_exc,TelemetryStageError):persist_exc=TelemetryStageError('PUBLISH',persist_exc)
            self._row['failure_evidence_durable']=False
            self._row['failure_artifact_error']=dict(failure_stage=persist_exc.stage,**self._exception_fields(persist_exc))
            print(json.dumps(dict(status='TELEMETRY_FAILURE_EVIDENCE_NOT_DURABLE',step_index=self.count,
                failure_stage=persist_exc.stage,complete=False)),file=sys.stderr,flush=True)

    def _joints(self):
        m,d,c=self.raw.model,self.raw.data,self.controller
        result=[]
        for i in range(m.njnt):
            qa=int(m.jnt_qposadr[i]);va=int(m.jnt_dofadr[i]);qn=int(m.jnt_qposadr[i+1]) if i+1<m.njnt else m.nq
            vn=int(m.jnt_dofadr[i+1]) if i+1<m.njnt else m.nv
            q=d.qpos[qa:qn].copy();bounds=m.jnt_range[i].copy()
            scalar=len(q)==1;limited=bool(m.jnt_limited[i])
            result.append(dict(id=i,name=m.joint(i).name,qpos_address=qa,qvel_address=va,
                qpos=q.tolist(),qvel=d.qvel[va:vn].copy().tolist(),limited=limited,range=bounds.tolist(),
                distance_lower=float(q[0]-bounds[0]) if scalar and limited else None,
                distance_upper=float(bounds[1]-q[0]) if scalar and limited else None,
                controller_arm_member=c.arm_qpos_adr<=qa<c.arm_qpos_adr+c.dof))
        return result

    def _contacts(self):
        m,d=self.raw.model,self.raw.data;result=[]
        for i in range(d.ncon):
            ct=d.contact[i];g1,g2=int(ct.geom1),int(ct.geom2)
            force=np.zeros(6,dtype=float);force_status='AVAILABLE'
            try:self.contact_force_reader(m,d,i,force)
            except Exception as e:force_status='UNAVAILABLE: '+repr(e);force=None
            frame=np.array(ct.frame,copy=True).reshape(3,3)
            result.append(dict(index=i,geom1_id=g1,geom1_name=m.geom(g1).name if g1>=0 else None,
                geom2_id=g2,geom2_name=m.geom(g2).name if g2>=0 else None,
                distance=float(ct.dist),normal=frame[0].tolist(),frame=frame.tolist(),position=np.array(ct.pos,copy=True).tolist(),
                contact_force_torque_local=copy_value(force),force_status=force_status,impulse=None,
                impulse_status='NOT_COMPUTED: instantaneous force is not an integrated impulse'))
        return result

    def _actuators(self):
        m,d=self.raw.model,self.raw.data
        # MuJoCo mjDSBL_CLAMPCTRL = 1<<7. Read named enum on real models.
        clamp_disabled=False
        if hasattr(m,'opt'):
            from mujoco import mjtDisableBit
            clamp_disabled=bool(int(m.opt.disableflags)&int(mjtDisableBit.mjDSBL_CLAMPCTRL))
        result=[]
        for i in range(m.nu):
            lo,hi=map(float,m.actuator_ctrlrange[i]);v=float(d.ctrl[i]);limited=bool(m.actuator_ctrllimited[i])
            result.append(dict(id=i,name=m.actuator(i).name,transmission_ids=copy_value(m.actuator_trnid[i]),
              ctrl=v,ctrlrange=[lo,hi],ctrllimited=limited,clamp_disabled=clamp_disabled,
              clip_flag=bool(limited and not clamp_disabled and (v<lo or v>hi)),
              effective_ctrl_inferred=float(np.clip(v,lo,hi)) if limited and not clamp_disabled else v,
              effective_ctrl_source='range inference; data.ctrl itself is not rewritten by observer'))
        return result

    def _snapshot(self):
        result=dict(qpos=copy_value(self.raw.data.qpos),qvel=copy_value(self.raw.data.qvel),ctrl=copy_value(self.raw.data.ctrl))
        try:result['joints']=self._joints()
        except Exception as exc:raise TelemetryStageError('JOINT_CAPTURE',exc) from exc
        try:result['actuators']=self._actuators()
        except Exception as exc:raise TelemetryStageError('ACTUATOR_CAPTURE',exc) from exc
        try:result['contacts']=self._contacts()
        except Exception as exc:raise TelemetryStageError('CONTACT_CAPTURE',exc) from exc
        return result

    def _profile(self,frame,event,arg):
        if event not in ('call','return'):return
        code=frame.f_code;local=frame.f_locals;c=self.controller
        try:
            if code==self.codes['solve_ik'] and local.get('self') is c:
                if event=='call':
                    self._row['ik']=dict(input_position=copy_value(local['target_pos']),input_quaternion=copy_value(local['target_quat']),
                        current_qpos=copy_value(self.raw.data.qpos[c.arm_qpos_adr:c.arm_qpos_adr+c.dof]),
                        current_qvel=copy_value(self.raw.data.qvel[c.arm_dof_adr:c.arm_dof_adr+c.dof]),
                        joint_limits=copy_value(c.joint_limits))
                else:self._row['ik']['q_target']=copy_value(arg)
            elif code==self.codes['least_squares'] and event=='return':
                # Only accept a solver called directly by this controller's solve_ik.
                if frame.f_back is None or frame.f_back.f_locals.get('self') is not c:return
                status=copy_value(local.get('status'))
                self._row['solver']=dict(status=status,success=(status in ('G_TOL','DX_TOL')) if status is not None else None,
                    success_meaning='solver tolerance status, not insertion or pose acceptance',
                    cost=copy_value(local.get('yfinal')),residual=copy_value(local.get('r')),
                    nfev=copy_value(local.get('n_res')),njev=copy_value(local.get('n_jac')),
                    returned=copy_value(arg),return_available=arg is not None)
            elif code in (self.codes['_ik_jacobian'],self.codes['compute_jacobian']) and local.get('self') is c and event=='return':
                self._row['jacobian_evaluations'].append(dict(kind='residual' if code==self.codes['_ik_jacobian'] else 'geometric',
                    q=copy_value(local.get('q')),value=copy_value(arg)))
            elif code==self.codes['compute_control'] and local.get('self') is c and event=='return':
                self._row['controller_output']=copy_value(arg)
            elif code==self.codes['_set_action'] and local.get('self') is self.raw:
                self._row['ctrl_before_write' if event=='call' else 'ctrl_after_write']=copy_value(self.raw.data.ctrl)
                if event=='call':self._row['absolute_action_after_env_clip']=copy_value(local['action'])
        except Exception as e:
            # Never throw into the solver's own exception/fallback path.
            self._row['observer_errors'].append(repr(e))
            stage='IK_CAPTURE' if code in (self.codes['solve_ik'],self.codes['least_squares'],self.codes['_ik_jacobian'],self.codes['compute_jacobian']) else 'ACTUATOR_CAPTURE'
            self._row.setdefault('_capture_failures',[]).append((stage,e,e.__traceback__))

    def step(self,action):
        if not self.enabled:return self.env.step(action)
        if sys.getprofile() is not None:raise RuntimeError('existing profiler: refuse to replace')
        self.count+=1
        self._row=dict(step=self.count,action=copy_value(action),jacobian_evaluations=[],observer_errors=[],_raw_step_available=False)
        try:self._row['pre']=self._snapshot()
        except TelemetryStageError as exc:
            self._record_failure(exc.stage,'NONE',exc,('pre',))
            raise exc.original
        error=None;error_tb=None;result=None
        try:
            sys.setprofile(self._profile)
            result=self.env.step(action)
        except BaseException as exc:
            error=exc;error_tb=exc.__traceback__
        finally:
            sys.setprofile(None)
        if error is not None:
            self._row['step_exception']=repr(error)
            self._record_failure('RAW_STEP','CONTACT_CAPTURE',error,self._missing_fields(self._row),error_tb)
            raise error.with_traceback(error_tb)
        self._row['_raw_step_available']=True;self._row['step_exception']=None
        # Post-processing only touches copied arrays, never controller/model state.
        try:
            self._row['post']=self._snapshot()
            try:
                for item in self._row['jacobian_evaluations']:
                    if item['value'] is None:continue
                    j=np.asarray(item['value'])
                    if item['kind']=='geometric':j=np.vstack(j)
                    item['singular_values']=self._singular_values(j)
                ik=self._row.get('ik',{})
                if ik.get('q_target') is not None:ik['q_target_minus_current_q']=(np.asarray(ik['q_target']).reshape(-1)-ik['current_qpos']).tolist()
            except Exception as exc:raise TelemetryStageError('POSTPROCESS',exc) from exc
            missing=self._missing_fields(self._row)
            if self._row.get('_capture_failures'):
                stage,exc,tb=self._row['_capture_failures'][0]
                self._record_failure(stage,'RAW_STEP',exc,missing,tb);return result
            if missing:
                exc=RuntimeError('missing telemetry fields: '+', '.join(missing))
                self._record_failure(self._stage_for_missing(missing),'RAW_STEP',exc,missing);return result
            self._row['complete']=not self._row['observer_errors']
            if not self._row['complete']:
                exc=RuntimeError('observer capture errors: '+'; '.join(self._row['observer_errors']))
                self._record_failure('POSTPROCESS','CONTACT_CAPTURE',exc,missing);return result
            row={key:value for key,value in self._row.items() if not key.startswith('_')}
            try:normalized=self._normalize_row(row)
            except Exception as exc:raise TelemetryStageError('JSON_NORMALIZE',exc) from exc
            try:payload=self._serialize_row(normalized)
            except Exception as exc:raise TelemetryStageError('SERIALIZE',exc) from exc
            self.last=row;self._row=row
            if self.stream:
                try:self._write_step_payload(payload)
                except Exception as exc:raise TelemetryStageError('WRITE',exc) from exc
                try:self._step_fsync()
                except Exception as exc:raise TelemetryStageError('FSYNC',exc) from exc
        except TelemetryStageError as exc:
            self._row['observer_errors'].append(repr(exc.original))
            self._record_failure(exc.stage,LAST_SUCCESSFUL_STAGE.get(exc.stage,'RAW_STEP'),exc,self._missing_fields(self._row))
        return result

    def close(self):
        if self.stream:self.stream.close();self.stream=None
