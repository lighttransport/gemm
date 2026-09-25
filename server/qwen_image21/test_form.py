"""Browser-logic checks for the Qwen-Image 2.1 demo form.

The page is a single HTML file with an inline script, so these tests extract the
script and run it under node against a minimal DOM stub. That catches the class
of bug a Python test cannot see: a mistyped identifier, a size cap computed from
the wrong branch, a payload that silently drops a setting. The `plan()` helper is
kept free of DOM access precisely so its rules can be exercised directly.
"""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "web/qwen_image21.html"
NODE = shutil.which("node") or shutil.which("nodejs")

STUB = """
const ids={};
function mk(id){return {value:'',textContent:'',innerHTML:'',
  classList:{toggle(){},add(){},remove(){}},
  addEventListener(){},querySelector(){return mk('x')},querySelectorAll(){return []},
  options:[],max:0,disabled:false}};
const document={getElementById:id=>ids[id]||(ids[id]=mk(id)),
  querySelector:s=>s.includes('mode')?{value:'native'}:mk('s'),
  querySelectorAll:()=>[{addEventListener(){}}]};
const fetch=()=>Promise.reject(new Error('offline'));
"""


def script() -> str:
    return PAGE.read_text(encoding="utf-8").split("<script>")[1].split("</script>")[0]


def run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([NODE], input=body, text=True, capture_output=True)


@unittest.skipUnless(NODE, "node is not installed")
class QwenImage21FormLogicTest(unittest.TestCase):
    def test_the_page_script_parses_and_runs(self):
        result = run(STUB + script())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_gates_the_tiled_controls_and_the_size_cap(self):
        body = script()
        plan = body[body.index("function plan("):body.index("function sync()")]
        cases = [
            # name, args, expected
            ("no preset", ("", "cuda", "native", 2, 1024, 1024),
             {"fast": False, "tiled": True, "cap": 1024, "ok": False}),
            ("preset, upscale 1", ("low8", "cuda", "native", 1, 1024, 1024),
             {"fast": True, "tiled": False, "cap": 2048, "ok": True}),
            ("preset, upscale 2", ("low8", "cuda", "native", 2, 2048, 2048),
             {"fast": True, "tiled": True, "cap": 4096, "ok": True}),
            ("compare mode", ("low8", "cuda", "compare", 2, 2048, 2048),
             {"fast": False, "tiled": True, "cap": 1024, "ok": False}),
            ("rocm", ("low8", "rocm", "native", 1, 1024, 1024),
             {"fast": False, "tiled": False, "cap": 1024, "ok": False}),
            ("reference mode", ("low8", "cuda", "reference", 1, 1024, 1024),
             {"fast": False, "tiled": False, "cap": 1024, "ok": False}),
        ]
        program = (
            "const HEALTH={size_limit:{reference:1024,fast:2048,tiled:4096}};\n" + plan +
            "const cases=" + json.dumps([[c[0], list(c[1]), c[2]] for c in cases]) + ";\n"
            "let bad=0;\n"
            "for(const [name,args,want] of cases){\n"
            "  const got=plan(...args);\n"
            "  if(JSON.stringify(got)!==JSON.stringify(want)){bad++;console.log('FAIL',name,JSON.stringify(got));}\n"
            "}\n"
            # A health payload that has not arrived must not throw or lock the form.
            "const none=plan('low8','cuda','native',2,4096,4096,null);\n"
            "if(none.cap!==4096||!none.ok){bad++;console.log('FAIL no-health',JSON.stringify(none));}\n"
            "process.exit(bad?1:0);\n")
        result = run(program)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_payload_carries_every_tiling_field(self):
        body = script()
        for field in ("upscale", "base_steps", "tile_tokens", "tile_overlap",
                      "refine_strength", "refine_seed", "vae_tile",
                      "vae_tile_overlap", "vae_tile_bleed"):
            self.assertIn(field + ":", body, f"{field} is missing from the request payload")
        # The decode overlap and bleed are only accepted alongside an explicit
        # decode tile, so the form has to withhold them when the tile is blank.
        for field in ("vae_tile_overlap", "vae_tile_bleed"):
            self.assertIn(f"$('vae_tile').value.trim() ? num('{field}') : null", body)
        # Progress is opt-in from the page: it needs a job to attach to and the
        # runner's per-step timing, which is what --profile-steps asks for.
        self.assertIn("job, profile_steps: true", body)
        self.assertIn("startPolling(job, 250)", body)
        self.assertIn("stopPolling()", body)

    def test_the_form_has_a_control_for_each_setting(self):
        markup = PAGE.read_text(encoding="utf-8")
        for control in ("upscale", "refine_strength", "base_steps", "refine_seed",
                        "tile_tokens", "tile_overlap", "vae_tile",
                        "vae_tile_overlap", "vae_tile_bleed"):
            self.assertRegex(markup, rf'id="{control}"', f"no form control named {control}")

    def test_the_progress_table_renders_a_step_row(self):
        """One real event must produce a row with the step, its own time and the
        running total; the display never invents a duration the runner did not
        report."""
        body = script()
        i = body.index("const fmtMs=")
        j = body.index("// The tiled refine needs")
        program = (
            "const $=id=>document.getElementById(id);\n"
            "const ids={};\n"
            "function mk(){return {value:'',textContent:'',innerHTML:'',hidden:false,style:{},children:[],"
            "removeChild(c){const i=this.children.indexOf(c);if(i>=0)this.children.splice(i,1)},"
            "classList:{toggle(){}},addEventListener(){},querySelector(){return mk()},options:[],"
            "append(c){this.children.push(c)}}}\n"
            "const document={getElementById:id=>ids[id]||(ids[id]=mk()),createElement:()=>mk()};\n"
            + body[i:j] +
            "applyProgress({stage:'denoise',accum_ms:3293.2,total_steps:3,elapsed_ms:13600,events:[\n"
            "  {kind:'step',stage:'denoise',index:1,total:8,sigma:0.5358,ms:1781.2,accum_ms:1781.2,elapsed_ms:12000},\n"
            "  {kind:'step',stage:'denoise',index:2,total:8,sigma:0.4,ms:1500.0,accum_ms:3281.2,elapsed_ms:13500},\n"
            "  {kind:'step',stage:'denoise',index:3,total:8,sigma:0.3,ms:null,accum_ms:3281.2,elapsed_ms:13600}]});\n"
            "const rows=document.getElementById('psteps').children;\n"
            "if(rows.length!==3){console.log('FAIL rows',rows.length);process.exit(1)}\n"
            "if(!/1\\/8/.test(rows[0].innerHTML)||!/1.78 s/.test(rows[0].innerHTML))"
            "{console.log('FAIL first row',rows[0].innerHTML);process.exit(1)}\n"
            # The accumulated column is the sum the denoiser reported, formatted.
            "if(!/3.28 s/.test(rows[1].innerHTML)){console.log('FAIL cum',rows[1].innerHTML);process.exit(1)}\n"
            # A step with no measured duration shows a placeholder, never a guess.
            "if(!/--/.test(rows[2].innerHTML)){console.log('FAIL unmeasured',rows[2].innerHTML);process.exit(1)}\n"
            "console.log('progress row rendering ok');\n")
        result = run(program)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("progress row rendering ok", result.stdout)

    def test_the_run_log_is_rendered_and_escaped(self):
        body = script()
        self.assertIn("data.log", body)
        self.assertIn(".log", PAGE.read_text(encoding="utf-8"))
        # Runner output goes straight into innerHTML, so it has to be escaped.
        self.assertIn("data.log.map(esc)", body)
        self.assertIn("&amp;", body)
        # The caveat is styled by a rule that has to exist, or a CPU reference
        # reads as ordinary text.
        self.assertIn(".meta.warn{", PAGE.read_text(encoding="utf-8"))

    def test_the_escaper_actually_escapes(self):
        """The escaping is a claim about behaviour, so run it: a note or a log
        line carrying markup must not become markup."""
        body = script()
        i = body.index("const esc=")
        j = body.index("function render(")
        program = (
            "const $=id=>document.getElementById(id);\n"
            "const ids={};\n"
            "function mk(){return {value:'',textContent:'',innerHTML:'',hidden:false,style:{},children:[],"
            "removeChild(c){const i=this.children.indexOf(c);if(i>=0)this.children.splice(i,1)},"
            "classList:{toggle(){}},addEventListener(){},querySelector(){return mk()},options:[],"
            "append(c){this.children.push(c)}}\n}\n"
            "const document={getElementById:id=>ids[id]||(ids[id]=mk()),createElement:()=>mk()};\n"
            + body[i:j] +
            "const hostile='<img src=x onerror=alert(1)> & \"quoted\"';\n"
            "const out=esc(hostile);\n"
            "if(out.includes('<img')){console.log('FAIL raw tag',out);process.exit(1)}\n"
            "if(!out.includes('&lt;img')||!out.includes('&amp;')){console.log('FAIL',out);process.exit(1)}\n"
            "console.log('escape ok');\n")
        result = run(program)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("escape ok", result.stdout)

    def test_the_form_offers_a_pytorch_device(self):
        markup = PAGE.read_text(encoding="utf-8")
        self.assertRegex(markup, r'id="refdevice"')
        for device in ("cuda", "rocm", "cpu"):
            self.assertIn(f'<option value="{device}"', markup)
        body = script()
        self.assertIn("reference_device: $('refdevice').value", body)
        # A native run has no reference, so the row stays out of the way.
        self.assertIn("row.hidden=mode==='native'", body)
        # An option the health report says is not installed must not be
        # selectable, or the run comes back as a refusal instead of a picture.
        self.assertIn("option.disabled=!ok", body)

    def test_the_device_selector_follows_what_is_installed(self):
        """Only devices with a PyTorch build may be offered, and a selection that
        turns out to be gone has to move somewhere runnable rather than leaving
        the form pointed at a request the server will refuse."""
        body = script()
        i = body.index("// The reference runs on a PyTorch build per device")
        j = body.index("for(const id of ['backend'")
        program = (
            "const ids={};\n"
            "const $=id=>document.getElementById(id);\n"
            "function mk(id){let v=id==='refdevice'?'cuda':'';const o={get value(){return v},"
            "set value(x){v=x;o.selectedOptions=[o.options.find(y=>y.value===x)]},"
            "textContent:'',innerHTML:'',hidden:false,style:{},children:[],disabled:false,"
            "selectedOptions:[],removeChild(){},classList:{toggle(){}},addEventListener(){},"
            "querySelector(){return mk('x')},options:[{value:'cuda',disabled:false},"
            "{value:'rocm',disabled:false},{value:'cpu',disabled:false}]};"
            "o.selectedOptions=[o.options[0]];return o}\n"
            "const document={getElementById:id=>ids[id]||(ids[id]=mk(id)),createElement:()=>mk('t'),"
            "querySelector:()=>({value:'reference'}),querySelectorAll:()=>[{addEventListener(){}}]};\n"
            "const fetch=()=>Promise.reject(new Error('offline'));\n"
            "const crypto={randomUUID:()=>'x'};const setTimeout=()=>0,clearTimeout=()=>{};\n"
            "let HEALTH=null;\n"
            + body[i:j] +
            "const detail={cuda:{torch:'2.14.0+cu130'},rocm:{torch:null},cpu:{torch:'2.14.0+cu130'}};\n"
            "HEALTH={reference:{cuda:true,rocm:false,cpu:true},reference_detail:detail};\n"
            # A native run has no reference, so the row stays out of the way.
            "syncRefDevice('native');\n"
            "if(!$('refdevice-row').hidden){console.log('FAIL shown for native');process.exit(1)}\n"
            "syncRefDevice('reference');\n"
            "if($('refdevice-row').hidden){console.log('FAIL hidden for reference');process.exit(1)}\n"
            # The device with no PyTorch build is not selectable.
            "const byValue=v=>$('refdevice').options.find(o=>o.value===v);\n"
            "if(!byValue('rocm').disabled){console.log('FAIL rocm selectable');process.exit(1)}\n"
            "if(!/no PyTorch build/.test(byValue('rocm').textContent))"
            "{console.log('FAIL no reason',byValue('rocm').textContent);process.exit(1)}\n"
            "if(byValue('cuda').disabled||byValue('cpu').disabled)"
            "{console.log('FAIL good device disabled');process.exit(1)}\n"
            "if(refLabel()!=='PyTorch CUDA'){console.log('FAIL label',refLabel());process.exit(1)}\n"
            # The CPU hint has to say it is not a parity result.
            "$('refdevice').value='cpu';syncRefDevice('compare');\n"
            "if(refLabel()!=='PyTorch CPU'){console.log('FAIL cpu label');process.exit(1)}\n"
            "if(!/not for numerical agreement/.test($('refdevice-hint').textContent))"
            "{console.log('FAIL cpu hint',$('refdevice-hint').textContent);process.exit(1)}\n"
            # A selection that has just become unavailable moves off itself.
            "$('refdevice').value='rocm';syncRefDevice('compare');\n"
            "if($('refdevice').selectedOptions[0].disabled)"
            "{console.log('FAIL stayed on a disabled device');process.exit(1)}\n"
            "console.log('device selector ok');\n")
        result = run(program)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("device selector ok", result.stdout)

    def test_no_stray_lookalike_identifiers(self):
        """A mistyped identifier in a minified one-liner fails at runtime, not at
        parse time, and only on the code path that reads it."""
        body = script()
        for name in ("tiled", "upscale", "limits", "preset", "backend"):
            declared = len(re.findall(rf"\b(?:const|let|var)\s+{name}\b", body))
            declared += len(re.findall(rf",\s*{name}\s*=", body))
            self.assertGreaterEqual(declared, 1, f"{name} is used but never declared")
        # 'tilted' arrived from a bad substitution once and broke the page only
        # when the tiled path ran.
        self.assertNotIn("tilted", body)


if __name__ == "__main__":
    unittest.main()
