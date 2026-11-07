import json,os,pathlib,time,subprocess,signal,urllib.request,urllib.parse,urllib.error,uuid,traceback,math,sys
prep=pathlib.Path('/artifacts/r2-runtime-preparation');root=pathlib.Path(os.environ['RUN_DIR']);out=root/'preflight';out.mkdir(exist_ok=False)
config=json.loads((prep/'commands.json').read_text());op=urllib.request.build_opener(urllib.request.ProxyHandler({}));record={'scope':'preflight_only_direct_plan_no_admission','source_revision':os.environ['IMPLEMENTATION_REVISION'],'started_unix_s':time.time(),'w3_requests':[],'actual_model_requests':0,'success':False};processes=[];handles=[]
def save(): (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
def request(url,body=None,timeout=30):
 req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'},method='POST' if body is not None else 'GET')
 try:
  with op.open(req,timeout=timeout) as response:return {'status':response.status,'headers':dict(response.headers.items()),'body':json.loads(response.read())}
 except urllib.error.HTTPError as e:return {'status':e.code,'body_text':e.read().decode()}
def start(name,argv,env):
 f=(out/(name+'.log')).open('xb');handles.append(f);p=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append((p,p.pid,name));return {'name':name,'pid':p.pid,'pgid':p.pid,'argv':argv}
def ready(url,kind):
 until=time.monotonic()+60;last=None
 while time.monotonic()<until:
  assert all(p.poll() is None for p,_,_ in processes),'owned process exited'
  try:
   last=request(url,timeout=3)
   if last['status']==200:
    if kind=='model':assert any(x['id']==os.environ['SERVED_MODEL_NAME'] for x in last['body']['data'])
    return last
  except (urllib.error.URLError,TimeoutError):pass
  time.sleep(.25)
 raise TimeoutError((url,last))
save()
try:
 for mode,port in [('prefill',37100),('decode',37110)]:
  for route in ['update_waiting_priority','reassign_waiting']:
   body={'request_id':'h20-w3-missing-'+uuid.uuid4().hex,'priority':0,'schedule_revision':1,'expected_revision':0}
   if route=='update_waiting_priority':body['queue_name']=None
   else:body.update(target_queue_name=mode+'_waiting',requeue_locally=False)
   result=request(f'http://127.0.0.1:{port}/hexgent/'+route,body);item={'mode':mode,'route':route,'request':body,'response':result};record['w3_requests'].append(item);save()
   assert result['status']==200,item
   assert result['body']['success'] is False and result['body']['results'],item
   assert all(not x['success'] and x['reason_code']=='not_found' for x in result['body']['results']),item
 record['w3_actual_business_handlers_passed']=True
 env=dict(os.environ);env.update(config['sidecar_env']);env['HEXGENT_REVISION_STATE_PATH']=str(out/'planner-smoke-revision')
 sidecar=list(config['sidecar_argv'])
 for flag,value in [('--capture-path',str(out/'events.jsonl')),('--telemetry-path',str(out/'planner-telemetry.jsonl'))]:sidecar[sidecar.index(flag)+1]=value
 record['processes']=[start('capture-sidecar',sidecar,env)];record['sidecar_ready']=ready('http://127.0.0.1:37140/health','health');save()
 gateway=list(config['gateway_argv']);gateway[gateway.index('--hexgent-planner-timeout-ms')+1]='10000';genv=dict(os.environ);genv.update(config['gateway_env']);record['processes'].append(start('direct-gateway',gateway,genv));record['gateway_ready']=ready('http://127.0.0.1:37150/v1/models','model');save()
 payload=json.loads((prep/'smoke-request.json').read_text());payload['stream']=False;payload.pop('stream_options',None);cal=json.loads(pathlib.Path('/inputs/runtime-inputs-v2/sharegpt.calibration.sanitized.runtime.jsonl').open().readline())['calls'][0];payload['hexgent'].update(cal['planner_hints']);payload['hexgent'].update(parent_call_ids=[],workflow_arrival_ms=int(time.time()*1000));(out/'sent-request.json').write_text(json.dumps(payload,indent=2)+'\n')
 record['actual_model_requests']=1;save();response=request(config['request_url'],payload,timeout=180);record['model_response']=response;save();assert response['status']==200,response
 usage=response['body']['usage'];assert usage['prompt_tokens']==39 and usage['completion_tokens']==8,usage
 headers={k.lower():v for k,v in response['headers'].items()};assert headers.get('x-hexgent-router-mode')=='hexgent',headers;assert int(headers['x-hexgent-schedule-revision'])>0,headers
 wf=config['workflow_id'];cid=config['call_id'];until=time.monotonic()+30;events=[]
 while time.monotonic()<until:
  p=out/'events.jsonl'
  if p.exists():events=[json.loads(json.loads(l)['event_body_utf8']) for l in p.read_text().splitlines() if l.strip()]
  matches=[e for e in events if e.get('event_type')=='call_completed' and e.get('workflow_id')==wf and e.get('call_id')==cid]
  if matches:break
  time.sleep(.25)
 assert len(matches)==1,events;event=matches[0];record['completion_event']=event
 timings={}
 for mode,port in [('prefill',37100),('decode',37110)]:
  result=request(f'http://127.0.0.1:{port}/hexgent/request_timing?'+urllib.parse.urlencode({'workflow_id':wf,'call_id':cid}));assert result['status']==200,result
  matched=[row[mode] for row in result['body']['timings'] if mode in row and row[mode]['workflow_id']==wf and row[mode]['call_id']==cid];assert len(matched)==1,result;timings[mode]=matched[0];record[mode+'_timing_response']=result
 p,d=timings['prefill'],timings['decode'];components=[p['prefill_service_prefix_ms'],p['prefill_transfer_tail_ms'],d['decode_execution_ms']];assert all(math.isfinite(x) and x>=0 for x in components)
 total=sum(components);measured=event['measured_service_ms'];assert measured is not None and (measured==1 if 0<total<.5 else abs(measured-total)<=.500001),(total,event)
 stage=event['payload']['stage_timing'];assert stage['measured_service_ms']==measured and 'transfer_ms' not in stage
 for key in ['prefill_service_prefix_ms','prefill_transfer_tail_ms']:assert abs(stage[key]-p[key])<.000001
 for key in ['decode_execution_ms','decode_pipeline_wait_ms']:assert abs(stage[key]-d[key])<.000001
 telemetry=[json.loads(l) for l in (out/'planner-telemetry.jsonl').read_text().splitlines()];assert any(x['endpoint']=='/schedule_request' and x.get('workflow_id')==wf and x.get('call_id')==cid and x['error'] is None for x in telemetry),telemetry
 record['timing_component_sum_ms']=total;record['actual_measured_service_ms']=measured;record['component_rounding_error_ms']=measured-total;record['r2_live_path_passed']=True;record['success']=True
except Exception as e:record['error']=repr(e);record['traceback']=traceback.format_exc()
finally:
 for p,pgid,name in reversed(processes):
  if p.poll() is None:
   assert os.getpgid(p.pid)==pgid;os.killpg(pgid,signal.SIGTERM)
   try:p.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(pgid,signal.SIGKILL);p.wait(timeout=2)
  record.setdefault('stopped_processes',[]).append({'name':name,'pid':p.pid,'exit_code':p.returncode})
 for h in handles:h.close()
 record['finished_unix_s']=time.time();save();print(json.dumps({k:v for k,v in record.items() if k in ['success','error','w3_actual_business_handlers_passed','r2_live_path_passed','timing_component_sum_ms','actual_measured_service_ms','component_rounding_error_ms','actual_model_requests']},indent=2))
sys.exit(0 if record['success'] else 2)
