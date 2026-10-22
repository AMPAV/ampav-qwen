# ampav-qwen

Qwen model tooling for the AMPAV environment.

Native Transformers is the default runtime. Qwen3-VL and Qwen3-Omni have
separate packages because their inputs and generation behavior
differ. OpenVINO GenAI is available as an explicitly selected Qwen3-VL option
for supported Intel systems.

```python
from ampav.qwen3vl import Qwen3VLTransformersVideo
from ampav.qwen3omni import Qwen3OmniTransformers
```

The public API remains under development.

## Development

Use the shared AMPAV virtual environment and install the package in editable
mode:

```bash
python -m pip install -e .
```

Install the Qwen3-Omni media helpers when using native audiovisual input:

```bash
python -m pip install -e ".[omni]"
```

Install OpenVINO only when selecting that Qwen3-VL runtime:

```bash
python -m pip install -e ".[openvino]"
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```
