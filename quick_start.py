#!/usr/bin/env python3
"""
EmberVLM Quick Start Example
Demonstrates basic model usage for robot selection and incident response.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import torch


def test_model_creation():
    """Test basic model creation."""
    print("=" * 50)
    print("EmberVLM Quick Start")
    print("=" * 50)

    from models import EmberVLM, EmberVLMConfig

    # Create model with default config
    print("\n1. Creating EmberVLM model...")
    config = EmberVLMConfig(
        vision_pretrained=False,  # Don't download pretrained for test
        vision_frozen=True,
        num_vision_tokens=8,
        hidden_size=768,
        num_layers=6,
        freeze_lm_layers=[0, 1, 2, 3],
        trainable_lm_layers=[4, 5]
    )

    model = EmberVLM(config)
    print("   Model created successfully!")

    # Print model info
    param_counts = model.count_parameters()
    print("\n2. Model Statistics:")
    for component, (total, trainable) in param_counts.items():
        print(f"   {component}: {total:,} total, {trainable:,} trainable")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (batch_size, 32))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    print(f"   Input image shape: {pixel_values.shape}")
    print(f"   Input text shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs['logits'].shape}")
    print("   Forward pass successful!")

    return model


def test_robot_selection():
    """Test robot selection dataset and prompts."""
    print("\n4. Testing Robot Selection...")

    from data import RobotSelectionDataset, get_robot_descriptions_string

    # Print robot descriptions
    print("   Available robots:")
    for line in get_robot_descriptions_string().split('\n')[:5]:
        if line.strip():
            print(f"   {line}")
    print("   ...")

    # Check if dataset exists
    dataset_path = Path("Multi-Robot-Selection/multi_robot_selection_dataset.json")
    if dataset_path.exists():
        dataset = RobotSelectionDataset(
            data_path=str(dataset_path),
            split="train",
            augment=False,
            max_samples=10
        )
        print(f"   Loaded {len(dataset)} robot selection samples")

        # Show first sample
        sample = dataset[0]
        print(f"   Sample task: {sample['task'][:50]}...")
        print(f"   Selected robots: {sample['selected_robots']}")
    else:
        print("   Dataset not found at expected path")

    print("   Robot selection test complete!")


def test_incident_response():
    """Test incident response functionality."""
    print("\n5. Testing Incident Response...")

    from data import INCIDENT_TYPES, ROBOT_RECOMMENDATIONS

    print("   Supported incident types:")
    for itype in list(INCIDENT_TYPES.keys())[:4]:
        rec = ROBOT_RECOMMENDATIONS.get(itype, {})
        print(f"   - {itype}: primary={rec.get('primary', 'N/A')}")
    print("   ...")

    print("   Incident response test complete!")


def test_pi_runtime():
    """Test Raspberry Pi inference runtime."""
    print("\n6. Testing Pi Runtime...")

    from deployment import EmberVLMPiRuntime

    runtime = EmberVLMPiRuntime(use_rules_fallback=True)

    # Test robot selection
    result = runtime.select_robot(
        task_description="Inspect a building exterior for damage"
    )

    print(f"   Task: Inspect building")
    print(f"   Selected: {', '.join(result['selected_robots'])}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   Inference time: {result['inference_time_ms']:.1f} ms")

    # Test incident response
    incident = runtime.plan_incident_response(
        incident_type="fire",
        description="Building fire in commercial district"
    )

    print(f"\n   Incident: {incident['incident_type']}")
    print(f"   Primary robot: {incident['primary_robot']}")

    print("   Pi runtime test complete!")


def main():
    """Run all quick start tests."""
    try:
        # Test model
        model = test_model_creation()

        # Test robot selection
        test_robot_selection()

        # Test incident response
        test_incident_response()

        # Test Pi runtime
        test_pi_runtime()

        print("\n" + "=" * 50)
        print("All tests passed! EmberVLM is ready to use.")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python scripts/train_all_stages.py --config configs/training.yaml")
        print("3. Deploy on Pi: python deployment/pi_inference.py --interactive")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to install dependencies:")
        print("  pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()

