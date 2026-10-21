from pathlib import Path
import json,html,textwrap,csv,shutil,zipfile,hashlib,gzip
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent;S=json.loads((P/'summary.json').read_text());D=json.loads(gzip.decompress((P/'data/analysis_inputs.json.gz').read_bytes()));B=S['matched_localization'];E=S['experiments'];blue='#245781'
def pct(x):return f'{100*x:.2f}%'
def ci(x):return f'{100*x[0]:.2f} to {100*x[1]:.2f}'
def money(x):return f'${x:.5f}'
def lab(cat):
 return {'fine_grained_action_localization':'Localization','fine_grained_how_recognition':'How recognition','fine_grained_why_recognition':'Why recognition','ingredient_ingredient_weight':'Ingredient weight'}[cat]
# Additional accuracy-cost figure requested in the mentor schedule.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f,a=plt.subplots(figsize=(7.2,3.5));a.plot([r['cost_per_32_usd'] for r in B],[r['accuracy']*100 for r in B],'o-',color=blue)
for r in B:a.annotate(f"size {r['size']}",(r['cost_per_32_usd'],r['accuracy']*100),xytext=(5,6),textcoords='offset points',fontsize=9)
a.set(xscale='log',xlabel='Estimated API cost per 32 questions (USD, log scale)',ylabel='Accuracy (%)',title='Localization accuracy versus cost',ylim=(23,39));a.set_xticks([.01,.02,.05,.1,.2,.4],['$0.01','$0.02','$0.05','$0.10','$0.20','$0.40']);a.xaxis.set_minor_locator(plt.NullLocator());a.set_xlim(.009,.43);a.grid(alpha=.16);f.text(.12,-.04,'Same 2,336 targets; actual token usage and cache discounts.',fontsize=8)
f.savefig(P/'figures/05_accuracy_vs_cost.png',dpi=250,bbox_inches='tight');f.savefig(P/'figures/05_accuracy_vs_cost.svg',bbox_inches='tight');plt.close(f)
# Content is kept as structured sections and exported to PDF and editable Markdown.
sections=[]
def add(title,paras=None,table=None,fig=None,caption=None):sections.append({'title':title,'paras':paras or [],'table':table,'fig':fig,'caption':caption})
add('Executive findings',[
'On the same 2,336 localization questions from 73 reports and 73 original videos, accuracy was 33.01%, 34.93%, 35.02%, 30.74%, and 26.67% at sizes 1, 4, 8, 16, and 32. The curve is not monotonically decreasing: performance is similar at sizes 4 and 8, then declines at 16 and 32.',
'The size 8 to size 32 difference is -8.35 percentage points (exploratory 95% paired original-video bootstrap interval: -11.04 to -5.65 points). This supports investigating a large-batch accuracy penalty in this prepared subset; it does not establish a universal effect across models, question families, or orderings.',
'All 4,355 effective API calls completed successfully. All 12,940 scored field outputs were valid option answers in complete JSON objects. Effective baseline cost was US$43.48539 for 148,526,221 tokens. These figures exclude superseded v1 results and earlier pilots.',
'E6 common-target accuracy was 41/96 (42.71%) with pure localization and 40/96 (41.67%) with mixed question types. A one-question difference provides little evidence of a composition effect in this run. Table scope: E1 uses the full prepared pool; E2-E5 use matched localization; E6 rows use common targets only.'
],table=[['Condition','Calls','Correct / scored','Accuracy','API cost USD']]+[[e.replace('E6_homogeneous','E6 pure').replace('E6_mixed','E6 mixed'),f"{v['calls']:,}",(f"{S['e6_common'][e]['correct']} / 96" if e.startswith('E6') else f"{v['correct']:,} / {v['fields']:,}"),(pct(S['e6_common'][e]['accuracy']) if e.startswith('E6') else pct(v['accuracy'])),f"{v['cost_usd']:.5f}"] for e,v in E.items()])
add('Scope and experimental design',[
'E1 covers all 3,212 questions in the prepared hd_epic_q40_v1 pool across 76 original videos. This is the full prepared input pool used in this project, not the complete official HD-EPIC VQA benchmark. E1 includes 2,861 localization, 175 how, 160 why, and 16 ingredient-weight questions.',
'E2 to E5 use 73 eligible reports with at least 32 localization questions. One fixed batch of 32 targets is selected per report. Size 4 makes eight calls, size 8 four calls, size 16 two calls, and size 32 one call for that same target set. The size 1 comparison filters E1 to those exact 2,336 report/field pairs; the E1 headline 35.55% covers a different population and is not used on the batch-size curve.',
'The selection seed is 20261017. There is one frozen question order, with smaller batches cut from the same ordering; three shuffles were intentionally not run. For each report the video window and prepared clip bytes are held fixed across conditions. All 73 shared clips use output-duration-v2.',
'E6 uses 24 reports, eight questions per call, and four common localization targets per report in positions 1 through 4 in both conditions. Pure reports add four localization companions; mixed reports add two how and two why companions. Only the common 96 targets support the paired comparison.'
])
add('Execution workflow and inference configuration',[
'Frozen questions and ground truth -> validate selection and source hashes -> assemble exact system/user prompts -> snapshot requests and call plan -> prepare and validate v2 clips -> send one API request per planned call -> checkpoint raw responses, usage, and timing -> parse JSON and score offline -> audit and aggregate matched results.',
'Model: gemini-3.5-flash-lite through the Vertex Express generateContent backend. Temperature: 0; max output tokens: 2,048; JSON response format. Top-p and generation seed were unset. Execution: concurrency 1, one attempt per call, 300-second request timeout. No automatic retries were used.',
'Local video preparation uses H.264, CRF 30, veryfast preset, 480-pixel height with aspect ratio preserved, 2 FPS, and no audio. The report\'s dataset-derived start/end window is passed to FFmpeg; duration is constrained on the output side. This window is shared by the report\'s calls, rather than cropping separately around each question\'s correct answer. The question text may also contain an original dataset segment or time-option range; that is distinct from the shared input clip window.',
'The canonical manifest stores source video, requested window, FPS, clip path, byte count, and SHA-256. The remaining-phase wrapper checks prepared bytes against the E5 manifest before HTTP for all shared reports. E1\'s three additional videos use the same v2 preparation rule.',
'Prepared inputs were uploaded as inline base64 video with text prompts. No ground-truth target values are passed as model-input fields. Ground truth is loaded only for offline scoring. The local request payload cap is 32,000,000 bytes.',
'Frozen pricing per million tokens: input US$0.30, cached input US$0.03, output US$2.50. Cost = (input - cached input) x input price + cached input x cached price + output x output price. Reasoning usage was zero; cached tokens are included in input and are not added again to total tokens.'
])
add('Localization accuracy versus report size',[
'Figure 1 compares precisely the same targets and video evidence. Confidence intervals resample original-video clusters rather than treating all questions as independent. There are 73 distinct source videos, each supplying 32 selected targets. We use 10,000 paired bootstrap resamples with analysis seed 20261020; intervals are percentile intervals.',
'Figure 1 intervals describe variation across these video clusters conditional on the frozen selection and one order. They do not estimate variability over repeated API runs, alternative question orders, dataset selections, or model versions.'
],fig='01_accuracy_vs_size.png',caption='Figure 1. Matched localization accuracy. Error bars are 95% original-video cluster bootstrap intervals.',table=[['Questions / call','Correct / 2,336','Accuracy','95% interval']]+[[str(r['size']),str(r['correct']),pct(r['accuracy']),ci(r['ci95'])+'%'] for r in B])
add('Paired changes and interpretation',[
'The size 4 and size 8 results differ by only two correct answers. The shape should therefore be described as a small-batch plateau followed by a decline at larger sizes, rather than a reliable optimum at exactly eight questions.',
'Exploratory paired intervals for the larger-batch comparisons remain below zero. The observed drops warrant follow-up on context interference, answer-position effects, and evidence allocation, but this baseline does not identify the mechanism. These comparisons are exploratory and not adjusted for multiple comparisons.',
'The existing HD-EPIC question construction can produce repeated action events and overlapping temporal alternatives. Frozen questions were retained rather than deduplicated, following the agreed scope. Answer-cue or event-overlap effects have therefore not been experimentally eliminated. Resampling videos addresses within-video dependence in the intervals, but does not remove cues between questions.'
],table=[['Comparison','Accuracy change pp','Paired 95% interval pp']]+[[f'{a} -> {b}',f"{S['paired_differences'][f'{a}_to_{b}']['delta']*100:+.2f}",ci(S['paired_differences'][f'{a}_to_{b}']['ci95'])] for a,b in [(1,4),(1,8),(1,16),(8,16),(8,32),(1,32)]])
add('Cost and tokens per completed set of 32 targets',[
'Normalization uses only the same 73 matched reports and 32 selected targets per report. In particular, size 1 cost is computed from the 2,336 matching E1 calls, not from all 3,212 E1 calls. Actual provider token usage and observed cache discounts are used.',
'Increasing batch size saves repeated video input. Size 8 costs approximately 86.1% less than size 1 per 32 targets while retaining similar observed accuracy. Size 32 is cheaper still, but its observed accuracy is 8.35 points lower than size 8. Cache hits were incidental rather than experimentally controlled; cost results are conditional on this run and pricing.'
],fig='02_cost_and_tokens.png',caption='Figure 2. Actual API cost and total tokens normalized to 32 matched localization targets.',table=[['Size','Calls / 32','USD / 32','Tokens / 32','All matched USD']]+[[str(r['size']),str(32//r['size']),f"{r['cost_per_32_usd']:.5f}",f"{r['tokens_per_32']:,.0f}",f"{r['cost_usd']:.5f}"] for r in B])
add('Accuracy versus cost',[
'This is a descriptive trade-off curve, not a deployment recommendation. Sizes 4 and 8 have nearly identical measured accuracy; their two-answer difference should not be treated as proof that size 8 is intrinsically more accurate. The large cost reduction comes primarily from avoiding repeated submission of the same video.',
'Effective E1-E6 cost is US$43.48539. The background E1-E4/E6 job accounts for US$42.50639; E5 all-v2 accounts for US$0.97899. Including the superseded v1 E5 successful calls adds US$0.79557, for US$44.28096 in this baseline-related ledger. Earlier smoke tests and pilots and Azure VM/storage charges are outside these totals. This is token-based estimated API cost, not a provider invoice.'
],fig='05_accuracy_vs_cost.png',caption='Figure 3. Matched accuracy against API cost per 32 targets. The cost axis is logarithmic.')
add('Question family performance with one question per call',[
'E1 pooled accuracy over the whole prepared pool is 1,142/3,212 (35.55%). Q2 primary estimates average per video subtype accuracy equally, as specified in the proposal. The older report used pooled question accuracy; this revision corrects that estimator. Localization is by far the largest family in this run. How and why recognition have higher observed accuracy than localization, but the question families differ in content and difficulty, so this is not a controlled causal effect of type alone.',
'Ingredient-weight results are particularly uncertain: only 16 questions across 10 source videos are available. The broad interval and small sample do not support a stable ranking. Localization here uses all 2,861 E1 localization questions; this differs from the 2,336 selected targets on the report-size curve.'
],fig='03_question_types.png',caption='Figure 4. Equal-video single-question E1 accuracy by original question family, with original-video cluster bootstrap intervals.',table=[['Question family','Pooled correct / fields','Video mean accuracy','95% interval']]+[[lab(r['category']),f"{r['correct']} / {r['fields']}",pct(r['video_weighted_accuracy']),ci(r['video_weighted_ci95'])+'%'] for r in S['question_types']])
add('Pure versus mixed question types on common targets',[
'Both conditions ask eight questions in each of 24 reports and retain the same four localization targets in the same positions. Pure localization scores 41/96 (42.71%), compared with 40/96 (41.67%) for mixed companions.',
'Among the 96 shared targets, five changed from correct to wrong and four from wrong to correct. The mixed-minus-pure difference is -1.04 percentage points; the paired original-video bootstrap 95% interval is -7.29 to +5.21 points. This run offers no clear evidence of a composition benefit or penalty. A narrow claim of equivalence is also not supported.',
'The all-192-field results are 39.58% pure and 50.00% mixed, but include different companion questions and question families. Those totals must not be used as the causal comparison of mixing. The paired common-target calculation avoids that mismatch.'
],fig='04_e6_paired.png',caption='Figure 5. Common-target accuracy under pure and mixed companions; whiskers are 95% video-bootstrap intervals.')
add('Latency usage and runtime',[
'The detached VM background batch ran from 2026-10-20 15:12:46 UTC to 21:17:04 UTC, about 6 hours 4 minutes. It completed despite local VPN changes. E5 was executed separately in two v2 runs, with 60 newly rerun calls and 13 retained v2 calls; their measured execution times are summed below and do not represent one contiguous E5 wall-clock interval.',
'API round-trip timing includes upload, network overhead, and provider response latency. It is not backend-only model inference time. Execution time additionally includes local preparation/cache access and pipeline work; source-hash preflight, detached-job overhead, and final export can lie outside the per-experiment execution timer.',
'The effective baseline uses 148,526,221 total tokens, including 5,429,320 cached input tokens within the input count. Reasoning tokens were zero. Larger batches take longer per API call but need fewer calls to complete 32 targets. All per-call timings and usage are exported in data/per_call.csv.'
],table=[['Condition','Mean API s','Median s','P95 s','Execution min']]+[[r['experiment'].replace('E6_homogeneous','E6 pure').replace('E6_mixed','E6 mixed'),f"{r['round_trip_mean_s']:.2f}",f"{r['round_trip_median_s']:.2f}",f"{r['round_trip_p95_s']:.2f}",f"{r['execution_wall_s']/60:.2f}"] for r in S['latency']])
add('Output validity scoring and hallucination limits',[
'All 4,355 effective calls have complete JSON objects with the required keys and no extra keys. All 12,940 scored outputs are legal option answers: no parse failures, missing answers, or invalid option values were found in the offline audit. The effective-call provider-failure count is zero.',
'Exact single-choice scoring compares each predicted option letter with the ground-truth target for that report ID and field ID. A valid but incorrect letter is scored wrong. An invalid or missing answer would also be wrong; transport/provider failures are tracked separately. The stored parser and scorer were rerun against all saved outputs, reproducing the persisted field scores.',
'A response such as "I think it is B" as the whole response violates the JSON contract. A sentence stored as a JSON value is not an allowed option letter. No such invalid outputs occurred in the effective runs. Actual prompts specify the exact expected field IDs and permitted option letters.',
'These multiple-choice results do not provide a verified semantic hallucination rate. An incorrect temporal option is an answer error, not sufficient evidence of hallucination. The observed structural invalid-answer rate is 0/12,940; it must not be reported as a 0% hallucination rate. A genuine hallucination figure would require a defined evidence-grounding rubric and an additional audit.'
])
add('Provenance audit and limits',[
'Offline checks passed for 4,355 unique effective calls and 12,940 experiment/call/field outcomes. Run snapshot fingerprints verify; every frozen model request matches the selection freeze exactly; recomputed predictions, outcomes, and correctness match saved evaluations; per-call costs sum to the reported effective ledger. Raw provider response text, input/output/cache/reasoning token counts, VIDEO modality usage, model version, and STOP finish reasons were checked against every normalized saved response.',
'Timing records confirm the same E5 v2 SHA-256 for every shared video in E1-E4/E6. There are 76 unique effective clip hashes across 76 source videos. The original source/video checks were sealed by the runner before execution; this report checks their saved artifacts and prepared-clip records. Large base64 wire payloads and video files remain on the VM rather than being duplicated in the report package.',
'Code provenance: production package ad1ca97. Fixed-scope runner artifacts for the v2 remainder and remaining baseline are preserved with their run plans and snapshots. These scripts specialize the existing runner rather than constituting a newly merged main-branch release. The earlier v1 E5 run had 10 crop-duration and three local-payload failures before HTTP. Those 13 reports were repaired with v2, and the other 60 were subsequently rerun with v2; original successes were preserved as history. The v1 output is excluded from the final accuracy curve.',
'Limitations: one model, one fixed order, one prepared dataset subset, potential correlated or mutually informative questions, no repeated-seed/model-run assessment, and sparse ingredient-weight data. Frozen price estimates and observed caches do not establish future costs. The confidence intervals are conditional and exploratory.'
])
add('Practical next steps',[
'The baseline is complete and can support a write-up describing the observed batch-size/cost trade-off. Use the exact matched localization curve as the primary result, and report the E6 common-target comparison rather than all-field mixed accuracy.',
'Before broader claims, test whether the large-batch decline persists under additional fixed orders or repeated calls, a second model, or an additional dataset. These are proposed follow-ups only; this report generation made no API requests. An explicit hallucination study needs an evidence-grounded definition and audited labels.',
'All source results remain immutable. Any follow-up should reuse the canonical v2 clip bytes, target IDs, and saved prompts when evaluating the same condition, and retain a distinct run identifier for changed inputs or settings.'
])
# Save exact sample material separately, without fabricated text.
ex=D['examples']['E1'];batch=D['examples']['E5'];(P/'data/example_exact_input_output.json').write_text(json.dumps(ex,ensure_ascii=False,indent=2));(P/'data/example_batch32_output.json').write_text(json.dumps(batch['response'],ensure_ascii=False,indent=2))
add('Actual input and output example',[
'The following is a saved E1 call, not a newly generated example. Report hdepic-dense-001 asks loc_445 using the shared v2 input clip from 300.406 to 505.699 seconds of source video P09-20240622-154652.mp4. The prompt\'s "this clip" time references are relative to that prepared clip. The raw model reply was {"loc_445": "C"}; ground truth is E, so this is a valid JSON response scored incorrect.',
'The exact system and user prompts are reproduced below. Full call metadata, prepared-video hash, raw response, and offline score are in data/example_exact_input_output.json. Every effective call has the same detail in data/all_model_inputs_outputs.jsonl.gz.'
])
sections[-1]['code']=ex['request']['system']+'\nUSER\n'+ex['request']['user']+'\nACTUAL RAW OUTPUT\n'+ex['response']['text']
add('Batch output and artifact index',[
'The following is the actual saved 32-question E5 JSON reply, formatted for readability. Its input contains the same question IDs as keys requested by the prompt; all 32 predictions are scored independently. Exact raw text is in data/example_batch32_output.json and the full input is in data/all_model_inputs_outputs.jsonl.gz.',
'Artifact directory: baseline_report_20261020. The English PDF is the reading copy; report.md is editable. Figures are supplied as PNG and SVG. data/per_field.csv contains all 12,940 predictions and ground truth; data/per_call.csv contains all 4,355 calls with tokens, cost, latency, clip hash, and source lineage.',
'data/matched_localization.csv, question_types.csv, and latency.csv are the plotted/summary tables. summary.json contains confidence intervals and paired differences. data/audit.json records passed checks. data/selection_manifest.json and canonical_media_manifest.json preserve the frozen selection and video inputs.',
'logs/ contains the complete text inference/evaluation logs for each phase, including E5_combined.log. data/all_model_inputs_outputs.jsonl.gz contains exact prompt text, raw saved model responses, usage/timing, prepared-video hash, and offline scores. It omits video base64 bytes. The source evidence archives remain in the adjacent baseline_remaining_v2_20261020, e5_all_v2_20261020, and e5_repair_20261019 directories.',
'VM source evidence: /home/azureuser/Capstone-14/outputs/hd_epic_baseline_20261017/remaining-v2, with separate all-v2 E5 runs under reruns/E5-all-v2-remaining60 and repairs/E5-output-duration-v2. Full wire payloads and videos are retained there. No credential values are included in this report package.'
])
sections[-1]['code']=json.dumps(json.loads(batch['response']['text']),indent=2)
# Markdown copy with the same sections, tables, and linked plot assets.
md=['# HD EPIC Baseline Experiment Report','\nPrepared 20 October 2026 | Gemini 3.5 Flash Lite | E1 to E6 | Final v2 results\n']
for sec in sections:
 md+=['## '+sec['title'],''];md += [x+'\n' for x in sec['paras']]
 if sec.get('fig'):md+=['!['+sec['caption']+'](figures/'+sec['fig']+')','',sec['caption'],'']
 if sec.get('table'):
  tb=sec['table'];md+=[' | '.join(tb[0]),' | '.join(['---']*len(tb[0]))]+[' | '.join(row) for row in tb[1:]]+['']
 if sec.get('code'):md+=['```text',sec['code'],'```','']
(P/'report.md').write_text('\n'.join(md))
# Clean report with one content section per page; exact examples flow naturally.
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=9.2,leading=13.2,spaceAfter=9,textColor=colors.HexColor('#26323b')));styles.add(ParagraphStyle(name='CaptionReport',fontSize=8,leading=10,textColor=colors.HexColor('#5e6973'),spaceAfter=10));styles.add(ParagraphStyle(name='CodeReport',fontName='Courier',fontSize=7.2,leading=9,spaceAfter=8));styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=23;styles['Title'].leading=27;styles['Title'].textColor=colors.black;styles['Title'].spaceAfter=14;styles['Heading1'].fontSize=16;styles['Heading1'].leading=20;styles['Heading1'].spaceAfter=12;styles['Heading1'].textColor=colors.HexColor(blue)
story=[]
def para(s,style='BodyReport'):return Paragraph(html.escape(s),styles[style])
for idx,sec in enumerate(sections):
 if idx:story.append(PageBreak())
 if idx==0:
  story += [Paragraph('HD EPIC Baseline<br/>Experiment Report',styles['Title']),para('Question dataset hd_epic_q40_v1 | Revised 21 October 2026','CaptionReport')]
 story.append(Paragraph(html.escape(sec['title']),styles['Heading1']))
 for t in sec['paras']:story.append(para(t))
 if sec.get('fig'):
  filename=P/'figures'/sec['fig'];iw,ih=ImageReader(str(filename)).getSize();w=6.8*72;h=w*ih/iw;story += [Image(str(filename),width=w,height=h),Spacer(1,5),para(sec['caption'],'CaptionReport')]
 if sec.get('table'):
  data=[[para(str(v),'CaptionReport') for v in row] for row in sec['table']];n=len(data[0]);widths=[6.8*72/n]*n
  if n==4:widths=[145,110,100,134.6]
  tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef3')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#c4cdd5')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),2),('LEFTPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f8f9')])]))
  story += [Spacer(1,6),tab]
 if sec.get('code'):
  wrapped='\n'.join('\n'.join(textwrap.wrap(line,width=104,replace_whitespace=False,drop_whitespace=False) or ['']) for line in sec['code'].splitlines());story += [Spacer(1,6),Preformatted(wrapped,styles['CodeReport'])]
def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#66717b'));c.drawString(45,27,'HD EPIC | Question dataset hd_epic_q40_v1 | Revised 21 October 2026');c.drawRightString(567,27,str(d.page))
doc=SimpleDocTemplate(str(P/'report.pdf'),pagesize=(612,792),rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=44,title='HD EPIC Baseline Experiment Report',author='Capstone project team');doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('PDF AND MARKDOWN SAVED',len(sections),'sections')
# Standalone browser copy with embedded figure images and relative data/log links.
import base64
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>HD EPIC Baseline Experiment Report</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#26323b;max-width:1000px;margin:48px auto;padding:0 28px}h1{font-size:32px;line-height:1.2;color:#111}h2{font-size:23px;color:#245781;margin-top:52px}p{max-width:92ch}table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dce2e7}th{background:#e9eef3}img{width:100%;height:auto}figcaption{font-size:13px;color:#66717b}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:18px;font-size:12px;line-height:1.5}a{color:#245781}nav{background:#f4f6f8;padding:18px}section{break-inside:avoid-page}@media print{body{margin:0;font-size:10pt}nav{display:none}h2{break-before:page}}</style><h1>HD EPIC Baseline Experiment Report</h1><p>Question dataset hd_epic_q40_v1 | Revised 21 October 2026</p><nav><a href="report.pdf">PDF report</a> · <a href="report.md">Editable Markdown</a> · <a href="data/per_call.csv">Per-call CSV</a> · <a href="data/per_field.csv">Per-field CSV</a> · <a href="data/all_model_inputs_outputs.jsonl.gz">All inputs and outputs</a> · <a href="logs/">Complete logs</a></nav>']
for sec in sections:
 parts+=['<section><h2>'+html.escape(sec['title'])+'</h2>']
 parts+=['<p>'+html.escape(t)+'</p>' for t in sec['paras']]
 if sec.get('fig'):
  b=base64.b64encode((P/'figures'/sec['fig']).read_bytes()).decode();parts+=['<figure><img alt="'+html.escape(sec['caption'])+'" src="data:image/png;base64,'+b+'"><figcaption>'+html.escape(sec['caption'])+'</figcaption></figure>']
 if sec.get('table'):
  parts+=['<table>']
  for i,row in enumerate(sec['table']):
   tag='th' if i==0 else 'td';parts+=['<tr>'+''.join(f'<{tag}>'+html.escape(str(x))+f'</{tag}>' for x in row)+'</tr>']
  parts+=['</table>']
 if sec.get('code'):parts+=['<pre>'+html.escape(sec['code'])+'</pre>']
 parts+=['</section>']
parts+=['</html>'];(P/'report.html').write_text('\n'.join(parts))
