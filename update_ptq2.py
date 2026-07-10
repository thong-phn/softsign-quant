import sys
import re

with open("export_loso_tflite_ptq-new.py", "r") as f:
    content = f.read()

# Finding the fold loop block to replace
old_loop_block = """    requested = _parse_subject_selection(args.subjects)
    fold_subjects = requested if requested is not None else all_train_subjects

    if args.model_path is not None and len(fold_subjects) != 1:
        raise ValueError("--model-path can only be used with exactly one subject in --subjects.")

    axis_name = "per-channel" if args.per_channel_quant else "shared-axis"
    results_path = project_root / "log" / f"{dataset_tag}_ptq_results_loso_tflite_{args.quantization}_{axis_name}.txt"
    if args.model_path is not None:
        results_path = project_root / "log" / f"{dataset_tag}_ptq_results_loso_tflite_{args.quantization}_{axis_name}_single.txt"

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        f.write(f"TFLite PTQ Results - LOSO ({args.quantization}, {axis_name})\\n{'=' * 50}\\n\\n")

    device = torch.device("cpu")          # export & conversion run on CPU
    metrics_history = {c: {"acc": [], "f1": []} for c in PTQ_CONFIGS}

    #  fold loop 
    for val_subject in fold_subjects:
        print(f"\\n{'=' * 60}")
        print(f"  Fold – validation subject {val_subject}")
        print(f"{'=' * 60}")

        ckpt = _resolve_loso_checkpoint(
            dataset_name=args.dataset,
            project_root=project_root,
            val_subject=val_subject,
            quantization=args.quantization,
            per_channel_quant=args.per_channel_quant,
            model_path=args.model_path,
        )

        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue

        pt_model, in_channels, had_quant_params, input_preprocessor = _load_pytorch_model(
            model_path=ckpt,
            device=device,
            quantization=args.quantization,
            per_channel_quant=args.per_channel_quant,
            expected_num_channels=num_channels,
        )

        #  datasets 
        train_subjects = [s for s in all_train_subjects if s != val_subject]
        train_ds = _make_dataset(args.dataset, root_path, train_subjects, split="train")
        test_ds = _make_dataset(args.dataset, root_path, test_eval_subjects, split=test_split)

        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  [skip] Empty train/test dataset for this fold")
            continue

        # Determine input shape from the first sample
        sample_x, _ = train_ds[0]
        freq_bins = sample_x.shape[-1]
        input_shape = (1, in_channels, freq_bins)
        n_params = _count_parameters(pt_model)
        print(f"  Dataset: {args.dataset}")
        print(f"  Input shape for export: {input_shape}")
        print(f"  PyTorch model parameters: {n_params:,}")
        print(
            "  Export quant layer: disabled "
            f"(checkpoint had quant params: {'yes' if had_quant_params else 'no'})"
        )
        if input_preprocessor is not None:
            print("  Input preprocessing: external learned quant transform is enabled")

        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        #  export & convert once per fold 
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            onnx_path      = tmpdir / "model.onnx"
            saved_model_dir = tmpdir / "saved_model"

            _export_onnx_and_convert(pt_model, onnx_path, saved_model_dir, input_shape)

            rep_gen = _make_representative_gen(
                train_ds,
                n_samples=256,
                input_preprocessor=input_preprocessor,
            )

            # Log model info once per fold
            with open(results_path, "a") as f:
                f.write(
                    f"\\nFold {val_subject} | Checkpoint: {ckpt} | Params: {n_params:,}"
                    f" | Quant disabled for export: yes"
                    f" | External input preprocess: {'yes' if input_preprocessor is not None else 'no'}\\n"
                )

            #  quantize & eval each config 
            for cfg in PTQ_CONFIGS:
                print(f"\\n   {cfg} ")
                tflite_model, ops, macs = _convert_to_tflite(str(saved_model_dir), cfg, rep_gen)
                if tflite_model is None:
                    continue

                tflite_size_kb = len(tflite_model) / 1024

                # Save .tflite for inspection in a persistent location
                tflite_dir = project_root / "models" / "tflite"
                tflite_dir.mkdir(parents=True, exist_ok=True)
                tflite_out = tflite_dir / f"{dataset_tag}_val_{val_subject}_{cfg}.tflite"
                tflite_out.write_bytes(tflite_model)

                ops_str, macs_str = _format_ops_macs(ops, macs)
                print(f"  TFLite size: {tflite_size_kb:.1f} KB  |  OPs: {ops_str}  |  MACs: {macs_str}")

                acc, f1 = _evaluate_tflite(
                    tflite_model,
                    test_loader,
                    input_preprocessor=input_preprocessor,
                )
                print(f"  Result: Acc {acc:.2f}%  F1 {f1:.4f}")

                metrics_history[cfg]["acc"].append(acc)
                metrics_history[cfg]["f1"].append(f1)

                with open(results_path, "a") as f:
                    f.write(
                        f"  {cfg} | Acc: {acc:.2f}% | F1: {f1:.4f}"
                        f" | Size: {tflite_size_kb:.1f} KB"
                        f" | OPs: {ops_str} | MACs: {macs_str}\\n"
                    )"""

# Instead of replacing that massive exact string which might have minor space issues,
# we can use a regex to match from "requested = _parse_subject_selection(args.subjects)" 
# to the end of the file before "with open(results_path, "a") as f:" for the summary.

pattern = re.compile(r'    requested = _parse_subject_selection\(args\.subjects\).*?(?=    #  summary)', re.DOTALL)

new_loop_block = """    requested = _parse_subject_selection(args.subjects)
    test_folds = requested if requested is not None else all_subjects

    if args.model_path is not None and len(test_folds) != 1:
        raise ValueError("--model-path can only be used with exactly one subject in --subjects.")

    axis_name = "per-channel" if args.per_channel_quant else "shared-axis"
    results_path = project_root / "log" / f"{dataset_tag}_ptq_results_loso_tflite_{args.quantization}_{axis_name}.txt"
    if args.model_path is not None:
        results_path = project_root / "log" / f"{dataset_tag}_ptq_results_loso_tflite_{args.quantization}_{axis_name}_single.txt"

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        f.write(f"TFLite PTQ Results - LOSO ({args.quantization}, {axis_name})\\n{'=' * 50}\\n\\n")

    device = torch.device("cpu")          # export & conversion run on CPU
    metrics_history = {c: {"acc": [], "f1": []} for c in PTQ_CONFIGS}

    #  fold loop 
    for test_subject in test_folds:
        remaining_subjects = [s for s in all_subjects if s != test_subject]
        for val_subject in remaining_subjects:
            print(f"\\n{'=' * 60}")
            print(f"  Fold – Test Subject: {test_subject} | Validation Subject: {val_subject}")
            print(f"{'=' * 60}")

            ckpt = _resolve_loso_checkpoint(
                dataset_name=args.dataset,
                project_root=project_root,
                test_subject=test_subject,
                val_subject=val_subject,
                quantization=args.quantization,
                per_channel_quant=args.per_channel_quant,
                model_path=args.model_path,
            )

            if not ckpt.exists():
                print(f"  [skip] checkpoint not found: {ckpt}")
                continue

            pt_model, in_channels, had_quant_params, input_preprocessor = _load_pytorch_model(
                model_path=ckpt,
                device=device,
                quantization=args.quantization,
                per_channel_quant=args.per_channel_quant,
                expected_num_channels=num_channels,
            )

            #  datasets 
            train_subjects = [s for s in remaining_subjects if s != val_subject]
            train_ds = _make_dataset(args.dataset, root_path, train_subjects, split="train")
            test_ds = _make_dataset(args.dataset, root_path, [test_subject], split=test_split)

            if len(train_ds) == 0 or len(test_ds) == 0:
                print("  [skip] Empty train/test dataset for this fold")
                continue

            # Determine input shape from the first sample
            sample_x, _ = train_ds[0]
            freq_bins = sample_x.shape[-1]
            input_shape = (1, in_channels, freq_bins)
            n_params = _count_parameters(pt_model)
            print(f"  Dataset: {args.dataset}")
            print(f"  Input shape for export: {input_shape}")
            print(f"  PyTorch model parameters: {n_params:,}")
            print(
                "  Export quant layer: disabled "
                f"(checkpoint had quant params: {'yes' if had_quant_params else 'no'})"
            )
            if input_preprocessor is not None:
                print("  Input preprocessing: external learned quant transform is enabled")

            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

            #  export & convert once per fold 
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                onnx_path      = tmpdir / "model.onnx"
                saved_model_dir = tmpdir / "saved_model"

                _export_onnx_and_convert(pt_model, onnx_path, saved_model_dir, input_shape)

                rep_gen = _make_representative_gen(
                    train_ds,
                    n_samples=256,
                    input_preprocessor=input_preprocessor,
                )

                # Log model info once per fold
                with open(results_path, "a") as f:
                    f.write(
                        f"\\nFold Test {test_subject} Val {val_subject} | Checkpoint: {ckpt} | Params: {n_params:,}"
                        f" | Quant disabled for export: yes"
                        f" | External input preprocess: {'yes' if input_preprocessor is not None else 'no'}\\n"
                    )

                #  quantize & eval each config 
                for cfg in PTQ_CONFIGS:
                    print(f"\\n   {cfg} ")
                    tflite_model, ops, macs = _convert_to_tflite(str(saved_model_dir), cfg, rep_gen)
                    if tflite_model is None:
                        continue

                    tflite_size_kb = len(tflite_model) / 1024

                    # Save .tflite for inspection in a persistent location
                    tflite_dir = project_root / "models" / "tflite"
                    tflite_dir.mkdir(parents=True, exist_ok=True)
                    tflite_out = tflite_dir / f"{dataset_tag}_test_{test_subject}_val_{val_subject}_{cfg}.tflite"
                    tflite_out.write_bytes(tflite_model)

                    ops_str, macs_str = _format_ops_macs(ops, macs)
                    print(f"  TFLite size: {tflite_size_kb:.1f} KB  |  OPs: {ops_str}  |  MACs: {macs_str}")

                    acc, f1 = _evaluate_tflite(
                        tflite_model,
                        test_loader,
                        input_preprocessor=input_preprocessor,
                    )
                    print(f"  Result: Acc {acc:.2f}%  F1 {f1:.4f}")

                    metrics_history[cfg]["acc"].append(acc)
                    metrics_history[cfg]["f1"].append(f1)

                    with open(results_path, "a") as f:
                        f.write(
                            f"  {cfg} | Acc: {acc:.2f}% | F1: {f1:.4f}"
                            f" | Size: {tflite_size_kb:.1f} KB"
                            f" | OPs: {ops_str} | MACs: {macs_str}\\n"
                        )
"""

content = pattern.sub(new_loop_block, content)

with open("export_loso_tflite_ptq-new.py", "w") as f:
    f.write(content)

