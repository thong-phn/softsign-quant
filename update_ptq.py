import sys
import re

with open("export_loso_tflite_ptq-new.py", "r") as f:
    content = f.read()

# Replace dataset logic
dataset_logic_old = """    if args.dataset == "wear":
        root_path = project_root / "datasets" / "wear"
        all_train_subjects = list(range(18))
        test_eval_subjects = list(range(18, 24))
        test_split = None
        dataset_tag = "wear"
        num_channels = 3
    elif args.dataset == "uci-har":
        root_path = project_root / "datasets" / "uci-har"
        all_train_subjects = _load_subject_ids(root_path / "train" / "subject_train.txt")
        test_eval_subjects = _load_subject_ids(root_path / "test" / "subject_test.txt")
        test_split = "test"
        dataset_tag = "uci-har"
        num_channels = 6
    elif args.dataset == "recgym":
        root_path = project_root / "datasets" / "recgym"
        all_train_subjects = list(range(1, 9))
        test_eval_subjects = [9, 10]
        test_split = None
        dataset_tag = "recgym"
        num_channels = 6
    elif args.dataset == "realworld":
        root_path = project_root / "datasets" / "realworld"
        all_train_subjects = list(range(1, 14))
        test_eval_subjects = [14, 15]
        test_split = None
        dataset_tag = "realworld"
        num_channels = 9
    else:  # mhealth
        root_path = project_root / "datasets" / "mheath"
        all_train_subjects = list(range(1, 10))
        test_eval_subjects = [10]
        test_split = None
        dataset_tag = "mhealth"
        num_channels = 9"""

dataset_logic_new = """    if args.dataset == "wear":
        root_path = project_root / "datasets" / "wear"
        all_subjects = list(range(24))
        test_split = None
        dataset_tag = "wear"
        num_channels = 3
    elif args.dataset == "uci-har":
        root_path = project_root / "datasets" / "uci-har"
        all_subjects = _load_subject_ids(root_path / "train" / "subject_train.txt") + _load_subject_ids(root_path / "test" / "subject_test.txt")
        test_split = "test"
        dataset_tag = "uci-har"
        num_channels = 6
    elif args.dataset == "recgym":
        root_path = project_root / "datasets" / "recgym"
        all_subjects = list(range(1, 11))
        test_split = None
        dataset_tag = "recgym"
        num_channels = 6
    elif args.dataset == "realworld":
        root_path = project_root / "datasets" / "realworld"
        all_subjects = list(range(1, 16))
        test_split = None
        dataset_tag = "realworld"
        num_channels = 9
    else:  # mhealth
        root_path = project_root / "datasets" / "mheath"
        all_subjects = list(range(1, 11))
        test_split = None
        dataset_tag = "mhealth"
        num_channels = 9"""

content = content.replace(dataset_logic_old, dataset_logic_new)

# Update _resolve_loso_checkpoint
checkpoint_old = """def _resolve_loso_checkpoint(
    dataset_name: str,
    project_root: Path,
    val_subject: int,
    quantization: str,
    per_channel_quant: bool,
    model_path: Path | None,
):
    if model_path is not None:
        return model_path

    if dataset_name == "uci-har":
        ds_prefix = "uci"
    else:
        ds_prefix = dataset_name
        
    prefix_parts = [f"{ds_prefix}_best_model_loso", quantization]
    if per_channel_quant:
        prefix_parts.append("per_channel")
    prefix = "_".join(prefix_parts)
    
    new_path = project_root / "models" / f"{prefix}_val_{val_subject}.pth"
    if new_path.exists():
        return new_path
        
    # Fallback for older models
    if dataset_name == "uci-har":
        old_prefix = "best_model_loso"
    else:
        old_prefix = "wear_best_model_loso"
        
    old_parts = [old_prefix, quantization]
    if per_channel_quant:
        old_parts.append("per_channel")
    return project_root / "models" / f"{'_'.join(old_parts)}_val_{val_subject}.pth\"\"\""""

# We'll use a regex to replace _resolve_loso_checkpoint
import re

cp_pattern = re.compile(r'def _resolve_loso_checkpoint.*?return project_root / "models" / f"\{\'\_\'\.join\(old_parts\)\}_val_\{val_subject\}\.pth"', re.DOTALL)
new_cp = """def _resolve_loso_checkpoint(
    dataset_name: str,
    project_root: Path,
    test_subject: int,
    val_subject: int,
    quantization: str,
    per_channel_quant: bool,
    model_path: Path | None,
):
    if model_path is not None:
        return model_path

    if dataset_name == "uci-har":
        ds_prefix = "uci"
    else:
        ds_prefix = dataset_name
        
    prefix_parts = [f"{ds_prefix}_best_model_loso", quantization]
    if per_channel_quant:
        prefix_parts.append("per_channel")
    prefix = "_".join(prefix_parts)
    
    # Try the new naming convention first
    new_path = project_root / "models" / f"{prefix}_test_{test_subject}_val_{val_subject}.pth"
    if new_path.exists():
        return new_path
        
    # Fallback for older models
    old_path = project_root / "models" / f"{prefix}_val_{val_subject}.pth"
    return old_path"""

content = cp_pattern.sub(new_cp, content)


with open("export_loso_tflite_ptq-new.py", "w") as f:
    f.write(content)

