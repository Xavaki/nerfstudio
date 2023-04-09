## MD features showcase

### SVGs
<object data="../diagrams/d2.svg" type="image/svg+xml"></object>

### html tags
<font color="#B58900">model</font>=<font color="#D33682"><b>VanillaModelConfig</b></font><b>(</b>
                    <font color="#B58900">_target</font>=&lt;class <font color="#859900">&apos;nerfstudio.models.vanilla_nerf.NeRFModel&apos;</font>&gt;,
                    <font color="#B58900">enable_collider</font>=<font color="#586E75"><i>True</i></font>,
                    <font color="#B58900">collider_params</font>=<b>{</b><font color="#859900">&apos;near_plane&apos;</font>: <font color="#2AA198"><b>2.0</b></font>, <font color="#859900">&apos;far_plane&apos;</font>: <font color="#2AA198"><b>6.0</b></font><b>}</b>,
                    <font color="#B58900">loss_coefficients</font>=<b>{</b><font color="#859900">&apos;rgb_loss_coarse&apos;</font>: <font color="#2AA198"><b>1.0</b></font>, <font color="#859900">&apos;rgb_loss_fine&apos;</font>: <font color="#2AA198"><b>1.0</b></font><b>}</b>,
                    <font color="#B58900">eval_num_rays_per_chunk</font>=<font color="#2AA198"><b>4096</b></font>,
                    <font color="#B58900">num_coarse_samples</font>=<font color="#2AA198"><b>64</b></font>,
                    <font color="#B58900">num_importance_samples</font>=<font color="#2AA198"><b>128</b></font>,
                    <font color="#B58900">enable_temporal_distortion</font>=<font color="#CB4B16"><i>False</i></font>,
                    <font color="#B58900">temporal_distortion_params</font>=<b>{</b><font color="#859900">&apos;kind&apos;</font>: &lt;TemporalDistortionKind.DNERF: <font color="#859900">&apos;dnerf&apos;</font>&gt;<b>}</b>
                <b>)</b>

### html iframes
<iframe src="./stdouts/vanilla-nerf.html"></iframe>


### python blocks
```python
# This is a Python code block
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return (fibonacci(n-1) + fibonacci(n-2))

print(fibonacci(10)) # prints the 10th Fibonacci number
```