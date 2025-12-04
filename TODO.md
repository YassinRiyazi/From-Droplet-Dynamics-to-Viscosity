- [] Test progressive precision decay
    Cosine annealing schedule for matmul precision

    ```python
    def set_precision(step, total_steps):
        # 1 → highest accuracy early, medium later
        t = step / total_steps

        if t < 0.3:
            mode = "highest"
        elif t < 0.7:
            mode = "high"
        else:
            mode = "medium"

        torch.set_float32_matmul_precision(mode)
        ```
    Then call once per training step:

    ```python
        for step in range(total_steps):
            set_precision(step, total_steps)
            loss = train_step(...)
    ```

    Combine with AMP (automatic mixed precision)

    You can still use:

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x)

    If you want anything smaller than "medium", you must explicitly use lower-precision dtypes (fp16, bf16, fp8 on Hopper).


    Anneal 1: FP32 matmul precision
        if step < warmup_steps:
            torch.set_float32_matmul_precision("highest")
        elif step < mid_steps:
            torch.set_float32_matmul_precision("high")
        else:
            torch.set_float32_matmul_precision("medium")   # lowest possible

    Anneal 2: Mixed precision (recommended)

    Start with BF16/FP16 only after model stabilizes:

        if step < stable_steps:
            autocast_dtype = torch.float32
        else:
            autocast_dtype = torch.bfloat16   # or float16

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            loss = model(input).loss