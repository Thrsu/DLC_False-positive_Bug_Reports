from openai import OpenAI
import pandas as pd
import time
import re

client = OpenAI(
    base_url='',
    api_key=''
)

input_file_path = "tvm_issue_with_example.xlsx"
output_file_path = "tvm_issue_fewshot_choice.xlsx"

df = pd.read_excel(input_file_path)

def get_openai_response(content, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert of TVM (a deep learning compiler)."},
                {"role": "user", "content": content}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI access fail，retrying: {e}")
        time.sleep(10)
        return None

responses = []
explanations = []

examples = {
    1: """
**Title**: [Bug] The entry value of attr should be integer. However Array is got  
**Description**: I got `TVMError: Check failed: (value != nullptr) is false: The entry value of attr 'tir_var_upper_bound' should be integer. However Array is got` when compiling a relax module.
```
FalsePositive_Probability: 0.95

Reasoning: The error is due to an incorrectly formatted attribute (Array instead of int), which likely stems from a user-side mistake and not from a compiler bug.
```
""",
    2: """
**Title**: Value None in attribute "mode" of operator Upsample is not valid  
**Description**: When converting an ONNX model containing the Upsample operator, TVM raises the following error:  
`tvm.error.OpAttributeInvalid: Value None in attribute "mode" of operator Upsample is not valid.`
```
FalsePositive_Probability: 0.05

Reasoning: The compiler fails to handle a missing attribute in a way consistent with the ONNX spec. This suggests a bug in the compiler, not user error.
```
""",
    3: """
**Title**: [Bug][Relax] LiftTransformParams leads to wrong inference results
**Description**:
The user observed that applying the `relax.transform.LiftTransformParams()` transformation resulted in significantly different inference results. This led to the assumption that the transform was buggy. A detailed reproduction script was provided, demonstrating how `np.testing.assert_allclose()` fails when comparing inference outputs before and after applying the transform.

However, it is because that the user failed to account for the function signature change introduced by `LiftTransformParams`. Specifically, the transformed parameters must be computed by calling the newly introduced `main_transform_params` function and passed into `main` accordingly. The original script neglected to do this, effectively skipping a preprocessing step and leading to incorrect results.

```
FalsePositive_Probability: 0.95

Reasoning: The issue stems from an incorrect test setup where the user misinterpreted the semantics of `LiftTransformParams`. The transform alters how parameters are passed by lifting computation into a separate function (`main_transform_params`). The user omitted this call in their updated pipeline, leading to incorrect inference results. There is no bug in the transformation itself—just a misuse. Thus, this is a classic example of a **false positive** bug report.
```
""",
    4: """
**Title**: AttributeError: module 'os' has no attribute 'setsid'
**Description**: When running the example `Auto-tuning a convolutional network for x86 CPU` from TVM on a Windows system, the following error occurs:

```
AttributeError: module 'os' has no attribute 'setsid'
```

This happens in the `tvm.rpc.server` module when it attempts to launch a subprocess with `preexec_fn=os.setsid`, which is a Unix-only call and does not exist on Windows. The failure prevents tuning from running entirely on Windows systems. The bug was resolved by conditionally using `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` on Windows instead of `preexec_fn=os.setsid`.

```
FalsePositive_Probability: 0.05

Reasoning: While the immediate error is due to a platform-specific incompatibility (i.e., `os.setsid` not existing on Windows), the root cause lies in TVM’s source code not handling cross-platform subprocess spawning correctly. This is not a user misconfiguration; rather, it's a missing platform guard in TVM's own implementation that makes the default RPC server behavior incompatible with Windows out of the box. Therefore, it is a real bug in TVM that must be fixed in its codebase.
```
""",
    5: """
**Title**: [Bug][Relax] build tvm from source failed relax\_vm/cuda/cuda\_graph\_builtin.cc
**Description**: When building TVM from the **unity** branch, the following compilation errors occurred in `cuda_graph_builtin.cc`:

```
error: ‘cudaGraphExec_t’ does not name a type
error: ‘cudaGraphLaunch’ was not declared in this scope
error: ‘cudaStreamBeginCapture’ was not declared in this scope
...
```

However, the **master** branch builds successfully on the same system.

Environment:

* TVM: unity branch (latest)
* OS: Ubuntu 18.04
* CUDA: 11.4
* GCC: 7.5
* CMake: 3.26
* LLVM: 12.0

The issue is caused by an outdated CUDA version (11.4), which lacks support for CUDA Graph APIs (introduced in CUDA 11.5+). Upgrading to CUDA 11.7 or newer resolves the problem.

```
FalsePositive_Probability: 0.95

Reasoning: Although the errors appear in TVM source code, they are due to missing symbols from the CUDA SDK caused by the user's outdated CUDA version (11.4). The correct fix is to upgrade the CUDA toolkit, not modify TVM's code. Therefore, this is a user environment misconfiguration, not a bug in TVM itself.
```
""",
    6: """
**Title**: Docker build for `demo_android` fails due to missing `TVM_VENV` environment variable
**Description**: Running the command `docker build -t tvm.demo_android -f docker/Dockerfile.demo_android ./docker` to build the `demo_android` Docker image fails at step 7 with the following error: ERROR: expect TVM_VENV env var to be set

This failure occurs because the script `ubuntu1804_install_python.sh` expects the `TVM_VENV` environment variable to be set. However, the `docker/build.sh` and the corresponding Dockerfile (`Dockerfile.demo_android`) do not set or export this required variable anywhere in the image build context. As a result, the default Docker build fails without user intervention.
```
FalsePositive_Probability: 0.05

Reasoning: While the immediate failure is due to a missing environment variable, the bug lies in TVM’s official Docker build scripts, which invoke a shell script that *requires* `TVM_VENV` without setting it or documenting it clearly for users. This creates an unusable default experience and is not a user misconfiguration. Therefore, it is a real bug in TVM’s Docker infrastructure that needs to be corrected in the repository itself.
```
""",
    7: """
**Title**: [Bug] NotImplementedError when converting pytorch module
**Description**:
I got the following error when trying to convert a PyTorch model to TVM:

```
NotImplementedError: The following operators are not implemented: ['prims::broadcast_in_dim', 'prims::mul', 'prims::squeeze', 'prims::split_dim', ..., 'aten::alias']
```

This occurred during the execution of `relay.frontend.from_pytorch(scripted_model, shape_list)`.

```
FalsePositive_Probability: 0.95

Reasoning: Although the error message lists many unsupported operators, the real issue stems from the user using `torch.jit.script` instead of `torch.jit.trace`, which is the correct API required by TVM's PyTorch frontend. Therefore, this is a user-side incorrect usage rather than a genuine bug in the compiler.
```
""",
    8: """
**Title**: [BUG][TUTORIAL] Tutorial for quantization need update 
**Description**: The TVM quantization tutorial fails when switching from data-aware to global scale quantization using the following line:

```python
with relay.quantize.qconfig(calibrate_mode='global', global_scale=8.0):
```

This leads to the following error:

```
ValueError: Unknown calibrate mode global
```

The issue arises because the tutorial incorrectly uses `calibrate_mode='global'`, while the correct calibration mode string is `'global_scale'`. The fix is simple—changing the argument string resolves the issue:

```diff
- with relay.quantize.qconfig(calibrate_mode='global', global_scale=8.0):
+ with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
```

```
FalsePositive_Probability: 0.05

Reasoning: Although the error appears to stem from incorrect usage of the API, the root cause is an outdated or incorrect official tutorial provided by the TVM project itself. Users following the documentation exactly as written encounter a failure due to an invalid parameter. This is not user misuse, but rather a documentation bug in the TVM codebase that required a fix in the tutorial. Therefore, it is a legitimate TVM bug.
```
"""
}

for i, row in df.iterrows():
    print(f"正在处理第 {i + 1} 行...")
    title = row["Title"]
    body = row["Body"]
    fp_id = row["FP_Example"]
    bug_id = row["Bug_Example"]
    # print(f"fp_id: {fp_id}, bug_id: {bug_id}")
    fp_example = examples.get(fp_id, "")
    bug_example = examples.get(bug_id, "")
    
    content = f"""
You are assisting in the triage of issue reports submitted by users of the deep learning compiler **TVM**. Each issue report contains a **Title** and a **Description**.

Your task is to analyze the issue and estimate the likelihood that the issue is a **FalsePositive** — meaning the problem is **not caused by a bug in the compiler**, but rather due to incorrect usage, invalid input, user environment misconfiguration, or misunderstanding of expected behavior. In contrast, if a issue is not a FalsePositive bug report, then it is a genuine bug in the deep learning compiler which was introduced by the compiler developers and must be fixed by modifying the compiler’s source code.

**Do not directly classify the issue. Instead, output the estimated probability (between 0.0 and 1.0) that the issue is a FalsePositive.**

---

## Important Notes
- Pay special attention to whether the error occurs in code or scripts maintained by OpenVINO: If the problematic line or configuration appears in OpenVINO’s official repository (e.g., scripts/setupvars.bat, cmake, .cpp, .py files), it is not a user mistake

- **Environment misconfiguration or compatibility issues should only be considered FalsePositive if there's strong evidence the issue is entirely on the user's side.**  
For example, a user setting an invalid path or missing dependency they were explicitly told to install.  
- However, if the misconfiguration is due to:
  - unclear documentation,
  - silently failing logic in TVM,
  - breaking changes introduced without version check,
  - or dependency incompatibility caused by TVM’s own setup/installation script,

then it is **not** a FalsePositive. It should be treated as a genuine bug in the compiler (FalsePositive_Probability closer to 0.0).

- When unsure whether an error is caused by the user or the system, **lean toward classifying it as TVM’s responsibility** unless there's direct evidence of user misuse.

---

## Examples
### Example 1:
{fp_example}

### Example 2:
{bug_example}
---

## Output Format

Please output your response in **exactly** the following format:

```
FalsePositive_Probability: <Float between 0.0 and 1.0>

Reasoning: <Brief explanation referencing the likely root cause and your rationale>
```

---
# Use this format to classify the issue below:
# - **Title**: {title}
# - **Description**: {body}
"""
    response = get_openai_response(content)
    explanations.append(response)

    confidence = None
    reasoning = ""

    if response:
        confidence_match = re.search(r"FalsePositive_Probability:\s*([0-9]*\.?[0-9]+)", response)
        reasoning_match = re.search(r"Reasoning:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

        if confidence_match:
            confidence = confidence_match.group(1)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

    df.at[i, "FalsePositive_Probability"] = confidence
    df.at[i, "Reasoning"] = reasoning
    df.at[i, "Explanation"] = explanations[-1]

    df.to_excel(output_file_path, index=False)

    time.sleep(3)