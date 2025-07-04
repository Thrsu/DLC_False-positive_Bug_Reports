from openai import OpenAI
import pandas as pd
import time
import re

client = OpenAI(
    base_url='',
    api_key=''
)

input_file_path = "tvm_discussion_with_example.xlsx"
output_file_path = "tvm_discussion_fewshot_choice.xlsx"

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
**Title**: Compilation error with composed Relay functions
**Description**: User defines two Relay functions add(x) = x + 1 and mul(y) = y * 2, composes them as combined(x) = mul(add(x)), and attempts to compile using relay.build(mod, target="llvm"). Compilation fails with:

TVMError: Check failed: f.defined() : primitive functions not set on Relay function by TECompiler

```
FalsePositive_Probability: 0.95

Reasoning: User-defined Relay functions were not annotated with `with_attr("Primitive", 1)`, which is required for the compiler to treat them as primitive functions. This is a usage error, not a compiler bug.
```
""",
    2: """
**Title**: Golang gotvm package build fails [solved]
**Description**: Trying to build the Golang gotvm package on Ubuntu 20.04 with Go 1.16.3 using the provided Makefile in tvm/golang/Makefile causes the following build error:
```
/dmlc/logging.h:132:10: error: #include expects “FILENAME” or 132 | #include DMLC_USE_LOGGING_LIBRARY
```
The root cause was identified as the Go bindings manually `#includee` all runtime source files instead of linking against libtvm_runtime.so. This causes fragile dependency handling that breaks whenever a new runtime source file is introduced.

```
FalsePositive_Probability: 0.05

Reasoning: The failure occurs due to the TVM-maintained Golang bindings using an incorrect and brittle method to link the runtime, which breaks with any internal changes to TVM. This is not due to user misuse or environment misconfiguration, but rather a design flaw in TVM’s own binding strategy, and must be fixed in TVM’s source code.
```
""",
    3: """
**Title**: `AssertionError: assert expr.struct_info.shape is not None`  
**Description**: When executing the `apply_rotary_emb_torch` function in `qwen2_vl_model.py`, the following error is thrown during the tensor multiplication `x * cos`:

```
AssertionError: assert expr.struct_info.shape is not None
```

The shapes involved are:
- `cos`: `[1, resized_height // 28 * (resized_width // 28) * 4, 16, 80]`
- `x`: `[1, resized_height // 14 * (resized_width // 14), 16, 80]`

The user believed the two shapes were equivalent since `resized_height` and `resized_width` were divisible by 28. However, the TVM developers clarified that expressions like `x // 2 * 2` are not treated as equal to `x` symbolically. The compiler cannot infer semantic equivalence between these shapes, hence `expr.struct_info.shape` remains `None`. The recommended fix is to use `match_cast` to explicitly unify the shape at runtime.

```
FalsePositive_Probability: 0.95

Reasoning: The error is caused by a mismatch in symbolic shapes due to user-provided expressions. Although the shapes may be numerically equal under certain assumptions, TVM requires exact symbolic matching or explicit runtime checks using `match_cast`. As explained by the TVM developers, this is expected behavior and not a compiler bug.
```
""",
    4: """
**Title**: TVM web build error, and unrecognised target
**Description**: User attempts to build TVM v0.10.0 with WebAssembly (wasm) and WebGPU support following the web README. During make, build fails due to:

error: integer value -1 is outside the valid range of values [0, 15] for this enumeration type
constexpr DLDeviceType kInvalidDeviceType = static_cast<DLDeviceType>(-1);
User suspects the issue may stem from their environment (Ubuntu 20.04 Docker container with Emscripten), and works around it by changing kInvalidDeviceType to a valid value. Later, runtime execution fails due to:

ValueError: Cannot recognize 'target'. Target creation from string failed: llvm -target=wasm32-unknown-unknown-wasm -system-lib

```
FalsePositive_Probability: 0.05

Reasoning: The build failure is caused by stricter enum checks in newer Clang versions, and the invalid cast exists in TVM source code. The later runtime failure suggests outdated or unsupported target string parsing. Both indicate TVM-side compatibility issues, not user misuse.
```
""",
    5: """
**Title**: Check failed: CUDA: misaligned address
**Description**: The user encountered a runtime error when using ProgramMeasurer:

Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading == false: CUDA: misaligned address
The error occurred during TVMArrayFree, while freeing an NDArray object.
System: Ubuntu 20.04, RTX 4090, CUDA 11.8.

```
FalsePositive_Probability: 0.95

Reasoning: This error is a false positive caused by asynchronous CUDA execution. Setting os.environ['CUDA_LAUNCH_BLOCKING'] = '1' serializes CUDA operations and resolves the issue, indicating the original error message was misleading due to race conditions in memory deallocation.
```
""",
    6: """
**Title**: [BYOC] Shared library size doubles when enabling BYOC with cuBLAS
**Description**:
The user encountered an issue when compiling an ONNX model using TVM with BYOC (Bring Your Own Codegen) targeting cuBLAS. After enabling cuBLAS BYOC, the size of the exported shared library (.so) approximately doubled compared to compilation without BYOC. The user confirmed that even when saving the components (lib, graph JSON, and params) separately, the total size remained larger. The issue did not occur when BYOC was disabled.

```
FalsePositive_Probability: 0.1
Reasoning:
This is a bug caused by duplication of constant parameters between the main GraphExecutorFactory module and the external cuBLAS module. The root cause is that cuBLAS’s external runtime module does not implement the get_const_vars function. Without this function, TVM cannot identify and de-duplicate constant parameters stored across both modules. The TVM core has a mechanism to avoid such duplication if external modules cooperate via the GetFunction("get_const_vars") interface. 
```
""",
    7: """
**Title**: [Unity] TVMScript can't copy multiple shape parameters from function
**Description**:
The user encountered an issue while defining a helper function get_unary_mod that dynamically constructs a TVMScript IRModule using multiple shape parameters (shape, o_shape). The TVMScript parser failed to recognize shape variables used solely in type annotations, resulting in an error:

error: Undefined variable: shape
```
FalsePositive_Probability: 0.95

Reasoning: This is a false positive caused by a limitation in how the TVMScript parser collects Python non-local variables. Python does not treat variables used only in type annotations as "nonlocal" for closures. Declaring the shape parameters as global variables works around the issue, confirming that the error was not due to incorrect logic but rather a scoping misinterpretation during script parsing.
```
""",
    8: """
**Title**: [PyTorch] TVM fails to compile DistilBERT QA model due to missing operator aten::masked_fill_
**Description**: The user attempted to compile a JIT-traced version of the distilbert-base-cased-distilled-squad model (fine-tuned for question answering) using TVM. While the model ran successfully outside TVM, converting it using relay.frontend.from_pytorch resulted in a failure due to a missing operator implementation. Specifically, the error reported was:

NotImplementedError: The following operators are not implemented: ['aten::masked_fill_']

```
FalsePositive_Probability: 0.05
Reasoning:
This is a valid bug caused by a missing operator (aten::masked_fill_) in TVM’s PyTorch frontend at the time. The in-place variant of masked_fill was not yet implemented, although the non-in-place version (masked_fill) was supported. 
```
"""
}

for i, row in df.iterrows():
    title = row["Title"]
    body = row["Body"]
    fp_id = row["FP_Example"]
    bug_id = row["Bug_Example"]
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
    print(f"第 {i + 1} 行已保存到 {output_file_path}")

    time.sleep(3)