"""
Dataset Integrity Checker
Identifies corrupted or invalid image files in the dataset
"""

from PIL import Image
from pathlib import Path
import json
from datetime import datetime

def check_dataset_integrity(dataset_root="dataset"):
    """
    Check all images in dataset for validity
    
    Args:
        dataset_root: Root directory of dataset
    
    Returns:
        Dictionary with integrity report
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_root}")
        return
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_checked": 0,
            "valid": 0,
            "invalid": 0
        },
        "by_split": {},
        "invalid_files": {}
    }
    
    # Check train, valid, test splits
    for split in ["train", "valid", "test"]:
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"⚠️  Split not found: {split}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Checking {split.upper()} split...")
        print(f"{'='*60}")
        
        split_summary = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "by_class": {}
        }
        
        # Check each class directory
        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_summary = {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "invalid_files": []
            }
            
            # Check all image files
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            
            for img_file in image_files:
                class_summary["total"] += 1
                results["summary"]["total_checked"] += 1
                
                try:
                    # Try to open and verify image
                    with Image.open(img_file) as img:
                        img.verify()
                    class_summary["valid"] += 1
                    results["summary"]["valid"] += 1
                    
                except Exception as e:
                    class_summary["invalid"] += 1
                    results["summary"]["invalid"] += 1
                    class_summary["invalid_files"].append({
                        "filename": img_file.name,
                        "error": str(e)
                    })
                    
                    # Store invalid file reference
                    if split not in results["invalid_files"]:
                        results["invalid_files"][split] = {}
                    if class_name not in results["invalid_files"][split]:
                        results["invalid_files"][split][class_name] = []
                    
                    results["invalid_files"][split][class_name].append({
                        "file": img_file.name,
                        "path": str(img_file),
                        "error": str(e)
                    })
            
            split_summary["by_class"][class_name] = class_summary
            split_summary["total"] += class_summary["total"]
            split_summary["valid"] += class_summary["valid"]
            split_summary["invalid"] += class_summary["invalid"]
            
            # Print class summary
            status_icon = "✅" if class_summary["invalid"] == 0 else "⚠️"
            print(f"\n{status_icon} {class_name:20} | Total: {class_summary['total']:4} | Valid: {class_summary['valid']:4} | Invalid: {class_summary['invalid']:4}")
            
            if class_summary["invalid"] > 0:
                for invalid_file in class_summary["invalid_files"][:3]:  # Show first 3
                    print(f"    └─ {invalid_file['filename']}: {invalid_file['error']}")
                if len(class_summary["invalid_files"]) > 3:
                    print(f"    └─ ... and {len(class_summary['invalid_files']) - 3} more")
        
        results["by_split"][split] = split_summary
        
        # Print split summary
        print(f"\n{'-'*60}")
        print(f"Split Summary ({split.upper()}):")
        print(f"  Total: {split_summary['total']}")
        print(f"  Valid: {split_summary['valid']}")
        print(f"  Invalid: {split_summary['invalid']}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total files checked: {results['summary']['total_checked']}")
    print(f"Valid files: {results['summary']['valid']}")
    print(f"Invalid files: {results['summary']['invalid']}")
    
    if results['summary']['invalid'] > 0:
        print(f"\n⚠️  Found {results['summary']['invalid']} invalid/corrupted files")
        print("\nInvalid files by split:")
        for split, classes in results["invalid_files"].items():
            print(f"\n  {split.upper()}:")
            for class_name, files in classes.items():
                print(f"    {class_name}: {len(files)} invalid files")
                for file_info in files[:2]:
                    print(f"      - {file_info['path']}")
                if len(files) > 2:
                    print(f"      ... and {len(files) - 2} more")
    else:
        print(f"\n✅ All files are valid and intact!")
    
    # Save report
    report_path = Path("dataset_integrity_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Full report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    check_dataset_integrity()
