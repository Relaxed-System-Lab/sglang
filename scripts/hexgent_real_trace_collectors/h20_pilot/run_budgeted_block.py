"""Root driver for the fixed frozen runner, with pair-wise budget checks only."""
import asyncio,hashlib,importlib.util,json,os,pathlib,time,types,traceback,sys
prep=pathlib.Path('/artifacts/r2-runtime-preparation');root=pathlib.Path(os.environ['RUN_DIR']);launch=json.loads(pathlib.Path('/artifacts/reference-r2-container-launch.json').read_text());status=json.loads((root/'status.json').read_text());deadline=min(launch['hard_deadline_unix_s'],status['started_unix_s']+float(os.environ['RUN_TIMEOUT_S']))-30
runner=prep/'run_paired_pilot.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='2ade58d47148ff1abe021d700671e5dbe6eaec9cfbd1f7c71c65a11aad724077'
spec=importlib.util.spec_from_file_location('frozen_pilot_runner',runner);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
args=types.SimpleNamespace(source_root=pathlib.Path('/src'),runtime_input_dir=pathlib.Path('/inputs/runtime-inputs-v2'),reference_dir=root/'reference-collection',run_dir=root/'paired-pilot',repeat=0,deadline_unix_s=deadline)
contract=m.validate_inputs(args);plan=m.prepare(args,contract);support=m.load_module('h20_run_support',prep/'h20_run_support.py');api=support.load_workflow_api(args.source_root);expected=m.read_json(os.environ['H20_EXPECTED_LIMITS_JSON']);transfer=m.read_json(os.environ['TRANSFER_PARAMETERS_JSON']);logs=json.loads(os.environ['H20_WORKER_LOG_PATHS_JSON']);refs={r['workflow_id']:r for r in map(json.loads,(args.reference_dir/'workflow-references.jsonl').read_text().splitlines())};records=[];checks=[]
def save(extra=None):
 value={'runs':records,'pair_budget_checks':checks,'all_runs_attempted':len(records)==8,'complete_balanced_block':len(records)==8 and all(x.get('valid_measurement') for x in records),'not_run_ordinals':[r['ordinal'] for r in plan['runs'] if r['ordinal'] not in {x['ordinal'] for x in records}],'effective_driver_deadline_unix_s':deadline,'approved_docker_hard_deadline_unix_s':launch['hard_deadline_unix_s'],'budget_policy':'Before each pair require remaining >= 1.25*2*(last planned arrival + max same-cell standalone latency)+60sec controller/cleanup allowance; heuristic guard, not runtime guarantee; no load/seed/gap changes.'}
 if extra:value.update(extra)
 m.save_json(args.run_dir/'block-results.json',value)
async def execute():
 for index in range(0,8,2):
  pair=plan['runs'][index:index+2];cell=next(c for c in plan['cells'] if c['dataset']==pair[0]['dataset'] and c['load']==pair[0]['load']);maxref=max(refs[w]['latency_ms'] for w in cell['workflow_ids'])/1000;estimate=1.25*2*(max(cell['arrival_offsets_s'])+maxref)+60;remaining=deadline-time.time();check={'pair_ordinals':[r['ordinal'] for r in pair],'remaining_s':remaining,'required_heuristic_s':estimate,'max_reference_s':maxref,'last_planned_arrival_s':max(cell['arrival_offsets_s']),'allowed':remaining>=estimate,'checked_unix_s':time.time()};checks.append(check);save()
  if not check['allowed']:save({'stopped_reason':'NOT_RUN_budget','unstarted_cell':[pair[0]['dataset'],pair[0]['load']]});return
  for run in pair:
   m.ensure_time(deadline);record=await m.execute_run(args,run,cell,support,api,expected,transfer,logs,contract['model']);records.append(record);save()
try:asyncio.run(execute())
except Exception as e:save({'error':repr(e),'traceback':traceback.format_exc()});raise
